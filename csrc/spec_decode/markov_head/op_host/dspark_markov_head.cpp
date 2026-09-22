/*
 * dspark_markov_head op_host — tiling computation + single fused launch.
 *
 * All kernel arguments are named lvalues (EXEC_KERNEL_CMD contract).
 * Workspaces are cached process-wide (the launch is asynchronous; a local
 * tensor could be freed before the kernel retires).
 */

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <vector>

#include <torch/extension.h>

#include "tiling/platform/platform_ascendc.h"
#include "adv_api/reduce/reduce_tiling.h"

#include "torch_kernel_helper.h"

#include "dspark_markov_head_kernels/aclrtlaunch_dspark_markov_head.h"

namespace ascend_kernel {
namespace {

constexpr int64_t B_PAD = 8;      // kernel-side padded batch row
constexpr int64_t R_FIXED = 256;  // markov rank (asserted below)

int64_t CeilDiv(const int64_t a, const int64_t b)
{
    return (a + b - 1) / b;
}

// Stripe width aligned to 16 elements (32B for fp16 rows) so that every
// per-core stripe base address and DataCopy blockLen stay 32B aligned.
int64_t AlignedStripe(const int64_t total, const int64_t cores)
{
    const int64_t raw = CeilDiv(total, cores);
    return CeilDiv(raw, 16) * 16;
}

struct WsCache {
    at::Tensor wsVal;
    at::Tensor wsIdx;
    at::Tensor scratch;
    at::Tensor idxBase;
    int64_t coreNum = 0;
    int64_t sw2 = 0;
    int64_t kPlus1 = 0;
};

WsCache &GetWs(const int64_t coreNum, const int64_t sw2, const int64_t kPlus1)
{
    static WsCache cache;
    if (cache.coreNum == coreNum && cache.sw2 == sw2 && cache.kPlus1 == kPlus1 &&
        cache.wsVal.defined()) {
        return cache;
    }
    auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1);
    cache.wsVal = at::empty({coreNum * 64}, opts);   // 2x128B lines per core: {v,i,0} + swapped {i,v,0}
    cache.wsIdx = cache.wsVal;                        // single interleaved workspace now
    cache.scratch = at::empty({kPlus1 * 32 + 256},
                              at::TensorOptions().dtype(at::kInt).device(at::kPrivateUse1));
    // idx base table [0 .. sw2) as fp32 (11KB GM), read once per core at init
    // P1 merge masks: 8 x 192 uint8 BIT-PACKED (Compare/Select TENSOR_SCALAR
    // mode maps bit L -> lane L over the 1536 dual-line lanes); row b byte
    // p*4 bit b covers lane p*32+b. Geometry constant, uploaded once.
    {
        auto m = at::zeros({8, 192}, at::TensorOptions().dtype(torch::kUInt8));
        auto byteIdx = at::arange(0, coreNum * 4, 4, at::TensorOptions().dtype(at::kLong));
        for (int64_t b = 0; b < 8; ++b) {
            m[b].index_put_({byteIdx}, at::full({coreNum}, 1 << b, at::TensorOptions().dtype(torch::kUInt8)));
        }
        cache.idxBase = m.contiguous().to(at::TensorOptions().dtype(torch::kUInt8).device(at::kPrivateUse1));
    }
    cache.coreNum = coreNum;
    cache.sw2 = sw2;
    cache.kPlus1 = kPlus1;
    return cache;
}

int64_t g_lastCoreNum = 48;
int64_t g_lastSw2 = 16;
int64_t g_lastK1 = 2;

}  // namespace

at::Tensor dspark_markov_head(const at::Tensor &logits, const at::Tensor &draft_token_ids,
                                  const at::Tensor &markov_w1_weight,
                                  const at::Tensor &markov_w2_weight)
{
    TORCH_CHECK(logits.scalar_type() == at::kHalf, "logits must be float16");
    TORCH_CHECK(draft_token_ids.scalar_type() == at::kLong, "draft_token_ids must be int64");
    TORCH_CHECK(markov_w1_weight.scalar_type() == at::kHalf, "w1 must be float16");
    TORCH_CHECK(markov_w2_weight.scalar_type() == at::kHalf, "w2 must be float16");
    TORCH_CHECK(logits.dim() == 3, "logits must be [B, K, V]");
    TORCH_CHECK(markov_w1_weight.dim() == 2 && markov_w1_weight.size(1) == R_FIXED,
                "w1 must be [V, 256]");
    TORCH_CHECK(logits.is_contiguous() && draft_token_ids.is_contiguous() &&
                    markov_w1_weight.is_contiguous() && markov_w2_weight.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t B = logits.size(0);
    const int64_t K = logits.size(1);
    const int64_t V = logits.size(2);
    const int64_t Vd = markov_w2_weight.size(0);
    TORCH_CHECK(draft_token_ids.size(0) == B && draft_token_ids.size(1) == K + 1,
                "draft_token_ids must be [B, K+1]");
    TORCH_CHECK(markov_w1_weight.size(0) == V, "w1 rows must equal V");
    TORCH_CHECK(Vd <= V, "Vd must not exceed V");
    TORCH_CHECK(B <= B_PAD, "B <= 7 supported (trace-calibrated)");

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();  // 48 on Ascend910B1
    static bool printedCores = false;
    if (!printedCores) {
        printedCores = true;
        printf("[dspark] GetCoreNumAiv=%lld GetCoreNumAic=%lld\n",
               static_cast<long long>(coreNum),
               static_cast<long long>(platform->GetCoreNumAic()));
        uint32_t mx = 0, mn = 0;
        {
            std::vector<int64_t> dims = {16, 256};
            const ge::Shape sh(dims);
            AscendC::GetReduceSumMaxMinTmpSize(sh, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                               true, false, mx, mn);
            printf("[dspark] ReduceSum AR {16,256} tmp max=%u min=%u bytes\n", mx, mn);
        }
        {
            std::vector<int64_t> dims = {8, 512};
            const ge::Shape sh(dims);
            AscendC::GetReduceMaxMaxMinTmpSize(sh, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                               true, false, mx, mn);
            printf("[dspark] ReduceMax AR {8,512} tmp max=%u min=%u bytes\n", mx, mn);
        }
        {
            std::vector<int64_t> dims = {8, 1024};
            const ge::Shape sh(dims);
            AscendC::GetReduceMaxMaxMinTmpSize(sh, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                               true, false, mx, mn);
            printf("[dspark] ReduceMax AR {8,1024} tmp max=%u min=%u bytes\n", mx, mn);
        }
        {
            std::vector<int64_t> dims = {8, 416};
            const ge::Shape sh(dims);
            AscendC::GetReduceMaxMaxMinTmpSize(sh, ge::DT_FLOAT, AscendC::ReducePattern::AR,
                                               true, false, mx, mn);
            printf("[dspark] ReduceMax AR {8,416} tmp max=%u min=%u bytes\n", mx, mn);
        }
        fflush(stdout);
    }
    const int64_t sw1 = AlignedStripe(Vd, coreNum);
    const int64_t sw2 = AlignedStripe(V - Vd, coreNum);

    WsCache &ws = GetWs(coreNum, sw2, K + 1);
    g_lastCoreNum = coreNum; g_lastSw2 = sw2; g_lastK1 = K + 1;
    at::Tensor out = at::empty_like(draft_token_ids);

    // EXEC_KERNEL_CMD lvalue contract: every tiling scalar is a named local.
    int64_t bArg = B;
    int64_t kArg = K;
    int64_t vArg = V;
    int64_t vdArg = Vd;
    int64_t sw1Arg = sw1;
    int64_t sw2Arg = sw2;
    int64_t coreNumArg = coreNum;
    int64_t stepKArg = 0;
    int64_t isFinalArg = 0;
    for (int64_t k = 0; k < K; ++k) {
        stepKArg = k;
        isFinalArg = 0;
        EXEC_KERNEL_CMD(dspark_markov_head, coreNum, logits, draft_token_ids, draft_token_ids,
                        markov_w1_weight, markov_w2_weight, out, ws.wsVal, ws.wsIdx, ws.scratch,
                        ws.idxBase, bArg, kArg, vArg, vdArg, sw1Arg, sw2Arg, coreNumArg,
                        stepKArg, isFinalArg);
    }
    isFinalArg = 1;
    EXEC_KERNEL_CMD(dspark_markov_head, coreNum, logits, draft_token_ids, draft_token_ids,
                    markov_w1_weight, markov_w2_weight, out, ws.wsVal, ws.wsIdx, ws.scratch,
                    ws.idxBase, bArg, kArg, vArg, vdArg, sw1Arg, sw2Arg, coreNumArg,
                    stepKArg, isFinalArg);
    return out;
}


std::vector<at::Tensor> dspark_markov_head_ws(const at::Tensor &dummy)
{
    (void)dummy;
    // returns clones of the most recent workspaces (updated by the last call;
    // if shapes changed since, the cache was reallocated accordingly)
    WsCache &c = GetWs(g_lastCoreNum, g_lastSw2, g_lastK1);
    return {c.wsVal.clone(), c.wsIdx.clone(), c.scratch.to(at::kFloat).clone()};
}

}  // namespace ascend_kernel

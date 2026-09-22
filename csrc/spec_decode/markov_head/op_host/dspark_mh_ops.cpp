/*
 * dspark_mh_ops op_host — L2/L3 multi-card single-step primitive wrappers.
 *
 * All persistent chain state (ws candidate buffer, tok history lines, emb /
 * bias buffers) is harness-allocated and passed in, so every wrapper is
 * stateless and the K-step chain interleaves HCCL collectives explicitly.
 * EXEC_KERNEL_CMD lvalue contract: every tiling scalar is a named local.
 */

#include <torch/extension.h>

#include <map>
#include <mutex>
#include <tuple>

#include "tiling/platform/platform_ascendc.h"

#include "torch_kernel_helper.h"

#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh2_argmax.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh2_bias.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh2_emb.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh3_argmax.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh3_emb.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh3_merge.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_probe_empty.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_probe_empty_fat.h"
#include "dspark_markov_head_kernels/aclrtlaunch_dspark_mh3_step.h"

namespace ascend_kernel {
namespace {

int64_t CeilDiv(const int64_t a, const int64_t b)
{
    return (a + b - 1) / b;
}

// Stripe width aligned so per-core GM publishes land on whole 128B lines.
int64_t AlignedStripe(const int64_t total, const int64_t cores, const int64_t align)
{
    return CeilDiv(CeilDiv(total, cores), align) * align;
}

void CheckLine16(const at::Tensor &t, const char *name)
{
    TORCH_CHECK(t.scalar_type() == at::kLong && t.dim() == 1 && t.size(0) == 16 &&
                    t.is_contiguous(),
                name, " must be a contiguous [16] int64 line");
}

void CheckWs(const at::Tensor &t, const at::Tensor &ref)
{
    TORCH_CHECK(t.scalar_type() == at::kLong && t.is_contiguous() && t.numel() == 48 * 16,
                "ws must be a contiguous [768] int64 candidate buffer (48 cores x 16 line");
    TORCH_CHECK(t.device() == ref.device(), "ws must live on the same device");
}

}  // namespace

// ---- L2 op A: [mode 1: merge prev ws -> tok] + mask_lookup -> emb_local ----
void dspark_mh2_emb(const at::Tensor &tok_line, const at::Tensor &ws, const at::Tensor &w1,
                    const at::Tensor &emb_out, const at::Tensor &tok_out, int64_t lo,
                    int64_t hi, int64_t mode)
{
    CheckLine16(tok_line, "tok_line");
    CheckWs(ws, tok_line);
    CheckLine16(tok_out, "tok_out");
    TORCH_CHECK(w1.scalar_type() == at::kHalf && w1.dim() == 2 && w1.size(1) == 256 &&
                    w1.is_contiguous(), "w1 must be contiguous [vs, 256] fp16");
    TORCH_CHECK(emb_out.scalar_type() == at::kHalf && emb_out.dim() == 2 &&
                    emb_out.size(1) == 256 && emb_out.is_contiguous(),
                "emb_out must be contiguous [B, 256] fp16");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();
    const int64_t B = emb_out.size(0);
    int64_t bArg = B;
    int64_t loArg = lo;
    int64_t hiArg = hi;
    int64_t coreNumArg = coreNum;
    int64_t modeArg = mode;
    int64_t oneBlock = 1;  // EXEC macro captures the blockdim token by name
    EXEC_KERNEL_CMD(dspark_mh2_emb, oneBlock, tok_line, ws, w1, emb_out, tok_out, bArg, loArg,
                    hiArg, coreNumArg, modeArg);
}

// ---- L2 op B: emb @ w2_shard^T -> bias_local [B, vs_pad] fp16 ----
void dspark_mh2_bias(const at::Tensor &emb, const at::Tensor &w2, const at::Tensor &bias_out)
{
    TORCH_CHECK(emb.scalar_type() == at::kHalf && emb.dim() == 2 && emb.size(1) == 256 &&
                    emb.is_contiguous(), "emb must be contiguous [B, 256] fp16");
    TORCH_CHECK(w2.scalar_type() == at::kHalf && w2.dim() == 2 && w2.size(1) == 256 &&
                    w2.is_contiguous(), "w2 must be contiguous [vs, 256] fp16");
    const int64_t B = emb.size(0);
    const int64_t vs = w2.size(0);
    const int64_t vsPad = bias_out.size(1);
    TORCH_CHECK(bias_out.scalar_type() == at::kHalf && bias_out.dim() == 2 &&
                    bias_out.size(0) == B && bias_out.is_contiguous(),
                "bias_out must be contiguous [B, vs_pad] fp16");
    TORCH_CHECK(vsPad % 64 == 0 && vsPad >= vs, "vs_pad must be a 64-multiple >= vs");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();
    int64_t bArg = B;
    int64_t vsArg = vs;
    int64_t vsPadArg = vsPad;
    int64_t swArg = AlignedStripe(vs, coreNum, 64);
    EXEC_KERNEL_CMD(dspark_mh2_bias, coreNum, emb, w2, bias_out, bArg, vsArg, vsPadArg, swArg);
}

// ---- L2 op C: (logits[:,k] + bias_full) first-max candidates -> ws ----
void dspark_mh2_argmax(const at::Tensor &logits, const at::Tensor &bias_flat,
                       const at::Tensor &ws, int64_t step_k)
{
    TORCH_CHECK(logits.scalar_type() == at::kHalf && logits.dim() == 3 &&
                    logits.is_contiguous(), "logits must be contiguous [B, K, V] fp16");
    TORCH_CHECK(bias_flat.scalar_type() == at::kHalf && bias_flat.dim() == 3 &&
                    bias_flat.is_contiguous(),
                "bias_flat must be contiguous [W, B, vs_pad] fp16 (all_gather output)");
    CheckWs(ws, logits);
    const int64_t B = logits.size(0);
    const int64_t K = logits.size(1);
    const int64_t V = logits.size(2);
    const int64_t Wn = bias_flat.size(0);
    const int64_t vsPad = bias_flat.size(2);
    TORCH_CHECK(bias_flat.size(1) == B, "bias_flat dim1 must equal B");
    TORCH_CHECK(V % Wn == 0, "V must be divisible by W");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();
    int64_t bArg = B;
    int64_t kArg = K;
    int64_t vArg = V;
    int64_t wnArg = Wn;
    int64_t vsArg = V / Wn;
    int64_t vsPadArg = vsPad;
    int64_t swArg = AlignedStripe(V, coreNum, 16);
    int64_t stepKArg = step_k;
    EXEC_KERNEL_CMD(dspark_mh2_argmax, coreNum, logits, bias_flat, ws, bArg, kArg, vArg, wnArg,
                    vsArg, vsPadArg, swArg, stepKArg);
}

// ---- L3 op D: full local step -> per-core packed keys in ws ----
// P0 perf iteration ("borrow the cube path"): the opD GEMV is delegated to
// aclnn MatMulV2 (emb @ w2.t()) — the SAME op the bf16 torch reference
// chain uses (fp32 accumulate -> bf16 RNE out = materialization
// point 1), measured 10.4-10.5us (fp16) at the W=4 shard shape vs ~196us
// for the fused in-kernel Cast+Mul+ReduceSum chain (probe_p0_route).
// Composition (signature unchanged; two AscendC launches + one aclnn op):
//   1. dspark_mh3_emb   : b-parallel merge of the gathered ws keys -> toks,
//                         w1 row gather, emb row publish (whole 128B lines)
//   2. at::matmul_out   : bias = emb @ w2.t()  (aclnn MatMulV2)
//   3. dspark_mh3_argmax: (bias + logits shard) point-2 bf16 RNE + first-max
//                         -> packed keys into this rank's ws_ag slot; block 0
//                         assembles tok_out from the emb launch's scratch
// The fused single-launch bf16 backup is kept as dspark_mh3_step_fused
// (A/B fallback; own GEMV accumulation order — verify per corpus).
namespace {

struct Mh3StepWorkspace
{
    at::Tensor emb;       // [B, 256] bf16
    at::Tensor bias;      // [B, vs] bf16 (matmul_out destination)
    at::Tensor tokScr;    // [B, 16] int64 full-line scratch
};

std::map<std::tuple<c10::DeviceIndex, int64_t, int64_t>, Mh3StepWorkspace> &Mh3StepWs()
{
    static std::map<std::tuple<c10::DeviceIndex, int64_t, int64_t>, Mh3StepWorkspace> cache;
    return cache;
}

std::mutex g_mh3_ws_mu;

// P2: the launch floor scales ~linearly with the block count (probe_p2_floor:
// ~1.4us + ~0.26us/block; buffer profile irrelevant). The argmax stripe work
// per core is B * ceil(stripe/1024) epilogue units (~0.7us each), so fewer
// blocks win when B is small. Deterministic in B alone so that every
// consumer (step argmax / emb merge scan count / final merge scan count)
// agrees. Slots [n, 48) of each rank's ws region are zero-filled by block 0
// of the argmax launch (key 0 is neutral under max).
int64_t Mh3ArgmaxBlocks(const int64_t B)
{
    return (B <= 2) ? 8 : 24;
}

}  // namespace

void dspark_mh3_step(const at::Tensor &logits, const at::Tensor &w1, const at::Tensor &w2,
                     const at::Tensor &ws_ag, const at::Tensor &seed_line,
                     const at::Tensor &tok_out, int64_t r0, int64_t world, int64_t step_k,
                     int64_t mode)
{
    // L3 chain is NATIVE bf16 only (fp16 retired): the GLM-5.2 dspark
    // deployment is bf16-locked and the fp16 cast workaround measurably
    // degraded draft acceptance (see docs/sub-agent-handover.md §8)
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 3 &&
                    logits.is_contiguous(), "logits must be contiguous [B, K, V] bf16");
    TORCH_CHECK(w1.scalar_type() == at::kBFloat16 && w1.dim() == 2 && w1.size(1) == 256 &&
                    w1.is_contiguous() && w1.size(0) == logits.size(2),
                "w1 must be the replicated contiguous [V, 256] bf16");
    TORCH_CHECK(w2.scalar_type() == at::kBFloat16 && w2.dim() == 2 && w2.size(1) == 256 &&
                    w2.is_contiguous(), "w2 must be contiguous [vs, 256] bf16");
    TORCH_CHECK(ws_ag.scalar_type() == at::kLong && ws_ag.is_contiguous() &&
                    ws_ag.numel() == world * 48 * 16,
                "ws_ag must be the contiguous [W,768] all_gather buffer");
    CheckLine16(seed_line, "seed_line");
    CheckLine16(tok_out, "tok_out");
    const int64_t B = logits.size(0);
    const int64_t K = logits.size(1);
    const int64_t V = logits.size(2);
    const int64_t vs = w2.size(0);
    TORCH_CHECK(r0 * vs + vs <= V, "rank shard [r0*vs, r0*vs+vs) must lie in [0, V)");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();

    // workspace (stable addresses -> capture/replay-safe; internal buffers
    // only, never alias user inputs/outputs; input tensors stay read-only)
    Mh3StepWorkspace *wsPtr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mh3_ws_mu);
        auto key = std::make_tuple(logits.device().index(), B, vs);
        auto &cache = Mh3StepWs();
        auto it = cache.find(key);
        if (it == cache.end()) {
            Mh3StepWorkspace w;
            auto opts = logits.options();
            w.emb = at::empty({B, 256}, opts);
            w.bias = at::empty({B, vs}, opts);
            w.tokScr = at::empty({B, 16}, ws_ag.options());
            it = cache.emplace(key, std::move(w)).first;
        }
        wsPtr = &it->second;
    }

    int64_t bArg = B;
    int64_t coreNumArg = coreNum;
    int64_t worldArg = world;
    int64_t modeArg = mode;
    // only blocks [0, B) do work — launch exactly B blocks (P2: the launch
    // floor scales with block count; B <= 8 < 48)
    EXEC_KERNEL_CMD(dspark_mh3_emb, bArg, ws_ag, seed_line, w1, wsPtr->emb, wsPtr->tokScr,
                    bArg, coreNumArg, worldArg, modeArg);
    // materialization point 1 (S2/S3): aclnn fp32-accumulate bf16-out GEMV,
    // the same op the bf16 torch deployment reference chain uses
    at::matmul_out(wsPtr->bias, wsPtr->emb, w2.t());

    const int64_t nBlocks = Mh3ArgmaxBlocks(B);
    int64_t kArg = K;
    int64_t vArg = V;
    int64_t vsArg = vs;
    int64_t r0Arg = r0;
    int64_t swArg = AlignedStripe(vs, nBlocks, 16);
    int64_t stepKArg = step_k;
    int64_t fillFromArg = nBlocks;
    EXEC_KERNEL_CMD(dspark_mh3_argmax, nBlocks, logits, wsPtr->bias, ws_ag, wsPtr->tokScr,
                    tok_out, bArg, kArg, vArg, vsArg, r0Arg, swArg, coreNumArg, stepKArg,
                    fillFromArg);
}

// A/B fallback: the fused single-launch opD (bf16; its in-kernel GEMV
// accumulation order differs from aclnnMatMulV2 — verify per corpus before
// trusting it as bit-exact backup)
void dspark_mh3_step_fused(const at::Tensor &logits, const at::Tensor &w1, const at::Tensor &w2,
                           const at::Tensor &ws_ag, const at::Tensor &seed_line,
                           const at::Tensor &tok_out, int64_t r0, int64_t world,
                           int64_t step_k, int64_t mode)
{
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 3 &&
                    logits.is_contiguous(), "logits must be contiguous [B, K, V] bf16");
    const int64_t B = logits.size(0);
    const int64_t K = logits.size(1);
    const int64_t V = logits.size(2);
    const int64_t vs = w2.size(0);
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();
    int64_t bArg = B;
    int64_t kArg = K;
    int64_t vArg = V;
    int64_t vsArg = vs;
    int64_t r0Arg = r0;
    int64_t swArg = AlignedStripe(vs, coreNum, 16);
    int64_t coreNumArg = coreNum;
    int64_t worldArg = world;
    int64_t stepKArg = step_k;
    int64_t modeArg = mode;
    EXEC_KERNEL_CMD(dspark_mh3_step, coreNum, logits, w1, w2, ws_ag, seed_line, tok_out, bArg,
                    kArg, vArg, vsArg, r0Arg, swArg, coreNumArg, worldArg, stepKArg, modeArg);
}

// ---- final-column merge: ws -> tok line (once per forward) ----
void dspark_mh3_merge(const at::Tensor &ws_ag, const at::Tensor &tok_out, int64_t B,
                      int64_t world)
{
    TORCH_CHECK(ws_ag.scalar_type() == at::kLong && ws_ag.is_contiguous() &&
                    ws_ag.numel() == world * 48 * 16,
                "ws_ag must be the contiguous [W,768] all_gather buffer");
    CheckLine16(tok_out, "tok_out");
    TORCH_CHECK(B >= 1 && B <= 8, "B must be in [1, 8]");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = platform->GetCoreNumAiv();
    int64_t bArg = B;
    int64_t coreNumArg = coreNum;
    int64_t worldArg = world;
    int64_t oneBlock = 1;
    EXEC_KERNEL_CMD(dspark_mh3_merge, oneBlock, ws_ag, tok_out, bArg, coreNumArg, worldArg);
}


// ---- P2 launch-floor probes (perf iteration only) ----
void dspark_probe_empty(const at::Tensor &out, int64_t nBlocks)
{
    TORCH_CHECK(out.scalar_type() == at::kLong && out.is_contiguous() &&
                    out.numel() >= nBlocks * 16,
                "probe out must be a contiguous int64 buffer >= nBlocks*16");
    int64_t blocksArg = nBlocks;
    EXEC_KERNEL_CMD(dspark_probe_empty, nBlocks, out, blocksArg);
}

void dspark_probe_empty_fat(const at::Tensor &out, int64_t nBlocks)
{
    TORCH_CHECK(out.scalar_type() == at::kLong && out.is_contiguous() &&
                    out.numel() >= nBlocks * 16,
                "probe out must be a contiguous int64 buffer >= nBlocks*16");
    int64_t blocksArg = nBlocks;
    EXEC_KERNEL_CMD(dspark_probe_empty_fat, nBlocks, out, blocksArg);
}

}  // namespace ascend_kernel

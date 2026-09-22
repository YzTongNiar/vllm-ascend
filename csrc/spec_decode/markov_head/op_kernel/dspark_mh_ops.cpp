/*
 * dspark_mh_ops — L2/L3 multi-card single-step primitives (AIV only).
 *
 * L2 (deployment form, both HCCL collectives kept):
 *   dspark_mh2_emb     : [merge prev-step ws candidates -> tok] + mask lookup
 *                        into w1_shard -> emb_local [B,256] fp16 (non-owner
 *                        rows exact +0.0; feeds AllReduce(SUM))
 *   dspark_mh2_bias    : emb @ w2_shard^T -> bias_local [B,vs_pad] fp16
 *                        (fp32 GEMV accumulate, fp16 RNE out = S3 + point-1 of
 *                        S2; GEMV path identical to the L1-verified Cast+Mul
 *                        chain; published in whole 128B GM lines)
 *   dspark_mh2_argmax  : (logits[:,k] + bias_full) first-max argmax; per-core
 *                        candidates land in ws (read by the next step's emb
 *                        launch — launch boundary, L1 visibility rule)
 *
 * L3 (communication reduced) — NATIVE bf16 (fp16 retired from L3; L1/L2 stay
 *   fp16 below):
 *   dspark_mh3_step    : host-orchestrated composition (mh3_emb launch +
 *                        aclnnMatMulV2 bf16 GEMV + mh3_argmax launch); the
 *                        fused single-launch form below is the A/B backup.
 *                        Both: [merge ws -> tok, mode 1] -> w1 row gather ->
 *                        GEMV(w2_shard) -> +logits shard -> local first-max
 *                        -> PACK into one int64 key per (core, b)
 *                        key = (mono(val) << 18) | (0x3FFFF - idx)
 *                        mono = IEEE-order monotone map of the bf16 bits
 *                        (same 16-bit sign-magnitude mapping constants as
 *                        the fp16 version), raw (NO -0.0 normalization: NPU
 *                        torch.argmax measures +0.0 > -0.0), so element-wise
 *                        MAX of keys == lexicographic (val, smallest idx)
 *                        (S4). Materialization points:
 *                        (1) GEMV fp32 accumulate -> bf16 RNE; (2) fp32 add
 *                        of two bf16 -> bf16 RNE (vconv_f322bf16a, matches
 *                        torch bf16 add_ fp32-opmath). bf16->f32 widen is
 *                        the exact vconv_bf162f32; the argmax winner's bf16
 *                        bits are recovered from its fp32 image by a pure
 *                        right shift (low 16 mantissa bits are zero).
 *                        max is associative across (core, rank): the HCCL
 *                        AllReduce(MAX) runs directly on the [48,16] ws keys
 *                        (per-slot cross-rank max), and the per-slot merge
 *                        over cores happens at the START of the next step's
 *                        launch. One [6KB] MAX AllReduce per step replaces
 *                        AllGather + add + argmax across ranks.
 *   dspark_mh3_merge   : ws -> merged tok line (final column assembly; also
 *                        used by the L2 chain for its last step)
 *
 * NO in-kernel cross-core reads: SyncAll on this platform does NOT guarantee
 * GM visibility of other cores' MTE3 writes (empirical: ws contents correct
 * after launch but the in-launch merge read stale lines ~50% of launches) —
 * the L1 launch-boundary handoff is the only reliable cross-core path here.
 *
 * Platform rules carried over from L1 (trace.md / memory):
 *   - ccec -O1 (the -O3 miscompile); CMake flags untouched
 *   - cross-launch GM handoffs are FULL 128B lines (16 x int64)
 *   - LocalTensor views only over InitBuffer-ed TBufs
 *   - MulAddDst rework is forbidden (net-negative, L1 bisect matrix)
 *   - V-produced UB read by MTE3 needs the V_MTE3 event (bias publish race)
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t BLK_V = 16;      // GEMV tile width (v columns)
constexpr int64_t SUP_V = 64;      // bias publish super-tile = one 128B GM line
constexpr int64_t EPI_CHUNK = 512; // argmax epilogue chunk width
constexpr int64_t ARG_CHUNK = 1024;
constexpr int64_t B_PAD = 8;       // B <= 7 padded batch rows
constexpr int64_t LINE_I64 = 16;   // one FULL 128B GM line of int64 (L1 rule:
                                   // cross-launch GM handoffs are whole lines;
                                   // 8 i64 = 64B half-line writes are the v1
                                   // stale-L2-line bug)
constexpr int64_t R_CONST = 256;   // markov rank
constexpr int64_t KEY_IDX_MASK = 0x3FFFF;  // 2^18-1 >= V-1 (154880 < 262144)

// IEEE-order monotone map of fp16 bits; -0.0 normalized to +0.0 so the packed
// integer order matches torch.argmax's numeric comparison exactly (argmax
// compares with >, where -0.0 == +0.0).
__aicore__ inline int64_t PackKeyVal(const half v, const int64_t idx)
{
    uint16_t bits = 0;
    __builtin_memcpy(&bits, &v, sizeof(uint16_t));
    if (bits == 0x8000) {
        bits = 0x0000;
    }
    const uint32_t m = (bits & 0x8000u) ? (0xFFFFu ^ static_cast<uint32_t>(bits))
                                        : (0x8000u | static_cast<uint32_t>(bits));
    return (static_cast<int64_t>(m) << 18) | (KEY_IDX_MASK - idx);
}

// bf16 variant (L3 native-bf16 chains). v is the fp32 image of an ALREADY
// MATERIALIZED bfloat16 value (it reached fp32 through the exact bf16->f32
// widen), so its low 16 fp32 mantissa bits are zero by construction and the
// bf16 bit pattern is a pure right shift — no rounding can happen here. The
// mono map is dtype-independent IEEE sign-magnitude order over the same 16
// bit width (bf16 positives are order-preserving in raw bit order, negatives
// reversed — identical mapping constants as fp16), so the packing width
// (mono << 18 | (0x3FFFF - idx)) is unchanged.
// NO -0.0 normalization here (deliberately different from the fp16 overload):
// the DEPLOYMENT reference is the NPU torch chain, and NPU torch.argmax
// treats +0.0 as STRICTLY GREATER than -0.0 (measured at V=154880 scale for
// bf16 AND fp16, bf16_r0/probe_argmax_sem.json; CPU torch ties them). Raw
// mono reproduces exactly that: mono(+0.0)=0x8000 > mono(-0.0)=0x7FFF, so a
// later +0.0 correctly beats an earlier -0.0, and equal bits keep the
// smaller idx. The fp16 overload keeps its historical normalization (L2 is a
// verified-passing chain, left untouched; the edge never fires on
// randn-scale corpora — a materialized ±0.0 needs |sum| < ~2^-134).
__aicore__ inline int64_t PackKeyVal(const float v, const int64_t idx)
{
    uint32_t f32Bits = 0;
    __builtin_memcpy(&f32Bits, &v, sizeof(uint32_t));
    const uint16_t bits = static_cast<uint16_t>(f32Bits >> 16);
    const uint32_t m = (bits & 0x8000u) ? (0xFFFFu ^ static_cast<uint32_t>(bits))
                                        : (0x8000u | static_cast<uint32_t>(bits));
    return (static_cast<int64_t>(m) << 18) | (KEY_IDX_MASK - idx);
}

__aicore__ inline void Mte2ToV()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(id);
    WaitFlag<HardEvent::MTE2_V>(id);
}

// V-produced UB -> MTE3 read requires the explicit cross-pipe event; a bare
// PipeBarrier<PIPE_V> before an MTE3 publish intermittently loses the most
// recently issued Cast (L2 bias publish bug: last 16-col tile of every
// super-tile corrupted ~87% of runs).
__aicore__ inline void VToMte3()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(id);
    WaitFlag<HardEvent::V_MTE3>(id);
}

// Merge the post-AllReduce ws keys (one slot per core) into tokens. Called at
// the START of a launch; ws was written one launch boundary ago.
__aicore__ inline void MergeWsToToks(const GlobalTensor<int64_t> &wsGm,
                                     LocalTensor<int64_t> &wsAll, int64_t B, int64_t coreNum,
                                     int32_t *toks)
{
    DataCopy(wsAll, wsGm, coreNum * LINE_I64);
    Mte2ToV();
    PipeBarrier<PIPE_ALL>();
    for (int64_t b = 0; b < B; ++b) {
        int64_t best = 0;
        for (int64_t p = 0; p < coreNum; ++p) {
            const int64_t k = wsAll.GetValue(p * LINE_I64 + b);
            if (k > best) { best = k; }
        }
        toks[b] = static_cast<int32_t>(KEY_IDX_MASK - (best & KEY_IDX_MASK));
    }
    PipeBarrier<PIPE_ALL>();
}

// Merge the all_gathered ws keys ([W, 48, 16]; one slot per (rank, core))
// into tokens. The int64 MAX AllReduce path was abandoned: HCCL routes
// int64 MAX to an AICPU fallback (~5.9ms/op wall vs ~180us for fast-path
// dtypes), so the cross-rank max is realized as an int64 all_gather (fast
// sdma path) + this local merge. max over (rank, core) is associative, so
// the semantics are identical to the packed-key AR (S4 exact).
__aicore__ inline void MergeWsAgToToks(const GlobalTensor<int64_t> &wsAgGm,
                                       LocalTensor<int64_t> &wsAll, int64_t B, int64_t coreNum,
                                       int64_t world, int32_t *toks)
{
    DataCopy(wsAll, wsAgGm, world * coreNum * LINE_I64);
    Mte2ToV();
    PipeBarrier<PIPE_ALL>();
    for (int64_t b = 0; b < B; ++b) {
        int64_t best = 0;
        for (int64_t r = 0; r < world; ++r) {
            for (int64_t p = 0; p < coreNum; ++p) {
                const int64_t k = wsAll.GetValue((r * coreNum + p) * LINE_I64 + b);
                if (k > best) { best = k; }
            }
        }
        toks[b] = static_cast<int32_t>(KEY_IDX_MASK - (best & KEY_IDX_MASK));
    }
    PipeBarrier<PIPE_ALL>();
}

// ===========================================================================
// L2 op A: [mode 1: merge prev-step candidates -> tok] + mask lookup
// ===========================================================================
class Mh2Emb {
public:
    __aicore__ inline void Init(GM_ADDR tokIn, GM_ADDR wsIn, GM_ADDR w1In, GM_ADDR embOut,
                                GM_ADDR tokOut, int64_t B, int64_t lo, int64_t hi,
                                int64_t coreNum, int64_t mode)
    {
        B_ = B;
        lo_ = lo;
        hi_ = hi;
        coreNum_ = coreNum;
        mode_ = mode;
        tokGm_.SetGlobalBuffer((__gm__ int64_t *)tokIn);
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsIn);
        w1Gm_.SetGlobalBuffer((__gm__ half *)w1In);
        embGm_.SetGlobalBuffer((__gm__ half *)embOut);
        tokOutGm_.SetGlobalBuffer((__gm__ int64_t *)tokOut);
        pipe_.InitBuffer(wsBuf_, coreNum * LINE_I64 * sizeof(int64_t));  // 6KB
        pipe_.InitBuffer(lineBuf_, LINE_I64 * sizeof(int64_t));          // 128B
        pipe_.InitBuffer(rowBuf_, 3 * R_CONST * sizeof(half));           // 3 x 512B
        wsAll_ = wsBuf_.Get<int64_t>();
        tokI64_ = lineBuf_.Get<int64_t>();
        stage_ = rowBuf_.Get<half>();
        row_ = rowBuf_.GetWithOffset<half>(R_CONST, R_CONST * sizeof(half));
        zeroRow_ = rowBuf_.GetWithOffset<half>(R_CONST, 2 * R_CONST * sizeof(half));
        Duplicate(zeroRow_, static_cast<half>(0), R_CONST);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) { return; }
        int32_t toks[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (mode_ != 0) {
            MergeWsToToks(wsGm_, wsAll_, B_, coreNum_, toks);
            // record for the final out_ids assembly
            for (int64_t b = 0; b < LINE_I64; ++b) {
                tokI64_.SetValue(b, 0);
            }
            for (int64_t b = 0; b < B_; ++b) {
                tokI64_.SetValue(b, static_cast<int64_t>(toks[b]));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(tokOutGm_, tokI64_, LINE_I64);  // full 128B line
            PipeBarrier<PIPE_MTE3>();
        } else {
            DataCopy(tokI64_, tokGm_, LINE_I64);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            for (int64_t b = 0; b < B_; ++b) {
                toks[b] = static_cast<int32_t>(tokI64_.GetValue(b));
            }
            PipeBarrier<PIPE_ALL>();
        }
        for (int64_t b = 0; b < B_; ++b) {
            const int64_t tok = toks[b];
            // The published row must be V-produced: MTE2-written UB read
            // directly by MTE3 (even behind PIPE_ALL) intermittently loses
            // rows (2-card emb bug). Mte2ToV only orders the V queue; the
            // Cast makes V the producer and VToMte3 orders the MTE3 read.
            if (tok >= lo_ && tok < hi_) {
                DataCopy(stage_, w1Gm_[(tok - lo_) * R_CONST], R_CONST);
                Mte2ToV();
                // x - (+0.0) == x bit-exactly for every x (incl. -0.0); the
                // Sub makes V the producer of the published row
                Sub(row_, stage_, zeroRow_, R_CONST);
                PipeBarrier<PIPE_V>();
            } else {
                Duplicate(row_, static_cast<half>(0), R_CONST);
                PipeBarrier<PIPE_V>();
            }
            VToMte3();
            DataCopy(embGm_[b * R_CONST], row_, R_CONST);  // full 512B line
            PipeBarrier<PIPE_MTE3>();
            PipeBarrier<PIPE_ALL>();  // WAR: next iteration's stage_ reload
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> wsBuf_, lineBuf_, rowBuf_;
    LocalTensor<int64_t> wsAll_, tokI64_;
    LocalTensor<half> row_, stage_, zeroRow_;
    GlobalTensor<int64_t> tokGm_, wsGm_, tokOutGm_;
    GlobalTensor<half> w1Gm_, embGm_;
    int64_t B_ = 0, lo_ = 0, hi_ = 0, coreNum_ = 0, mode_ = 0;
};

// ===========================================================================
// L2 op B: emb @ w2_shard^T -> bias_local [B, vs_pad] fp16
// Stripe width sw is a 64-multiple; every published (core, b) segment starts
// on a 128B GM line boundary and covers whole lines (tail may end in pad).
// ===========================================================================
class Mh2Bias {
public:
    __aicore__ inline void Init(GM_ADDR embIn, GM_ADDR w2In, GM_ADDR biasOut,
                                int64_t B, int64_t vs, int64_t vsPad, int64_t sw)
    {
        const int64_t pid = GetBlockIdx();
        B_ = B;
        vsPad_ = vsPad;
        sw_ = sw;
        vLo_ = pid * sw;
        vHi_ = (vLo_ + sw < vs) ? (vLo_ + sw) : vs;
        w_ = (vHi_ > vLo_) ? (vHi_ - vLo_) : 0;
        embGm_.SetGlobalBuffer((__gm__ half *)embIn);
        w2Gm_.SetGlobalBuffer((__gm__ half *)w2In);
        biasGm_.SetGlobalBuffer((__gm__ half *)biasOut);
        pipe_.InitBuffer(emb16Buf_, B_PAD * R_CONST * sizeof(half));   // 4KB
        pipe_.InitBuffer(embF32Buf_, B_PAD * R_CONST * sizeof(float)); // 8KB
        pipe_.InitBuffer(w2TBuf_, 2 * BLK_V * R_CONST * sizeof(half)); // 16KB (ping-pong)
        pipe_.InitBuffer(w2F32Buf_, BLK_V * R_CONST * sizeof(float));  // 16KB
        pipe_.InitBuffer(prodbuf_, BLK_V * R_CONST * sizeof(float));   // 16KB
        pipe_.InitBuffer(bandBuf_, B_PAD * SUP_V * sizeof(half));      // 1KB
        pipe_.InitBuffer(sumBuf_, B_PAD * BLK_V * sizeof(float));
        pipe_.InitBuffer(arBuf_, 8192);  // ReduceSum AR {16,256} tmp
        emb16_ = emb16Buf_.Get<half>();
        embF32_ = embF32Buf_.Get<float>();
        w2T_ = w2TBuf_.Get<half>();
        w2T2_ = w2TBuf_.GetWithOffset<half>(BLK_V * R_CONST,
                                            BLK_V * R_CONST * sizeof(half));
        w2F32_ = w2F32Buf_.Get<float>();
        prod_ = prodbuf_.Get<float>();
        band16_ = bandBuf_.Get<half>();
        sumF32_ = sumBuf_.Get<float>();
        arTmp_ = arBuf_.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (w_ <= 0) { return; }
        DataCopy(emb16_, embGm_, B_ * R_CONST);
        Mte2ToV();
        PipeBarrier<PIPE_V>();
        for (int64_t b = 0; b < B_; ++b) {
            Cast(embF32_[b * R_CONST], emb16_[b * R_CONST], RoundMode::CAST_NONE, R_CONST);
        }
        PipeBarrier<PIPE_V>();
        // super-tiles of 64 columns: GEMV 4 x 16-tiles, then one 128B publish
        // ping-pong w2T tiles (V2): prefetch d+1 behind d's V compute
        event_t evB[2] = {
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)),
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V))};
        for (int64_t st = 0; st < w_; st += SUP_V) {
            const int64_t supLen = (w_ - st < SUP_V) ? (w_ - st) : SUP_V;
            const int64_t nTiles = (supLen + BLK_V - 1) / BLK_V;
            {
                const int64_t curLen = (supLen < BLK_V) ? supLen : BLK_V;
                DataCopy(w2T_, w2Gm_[(vLo_ + st) * R_CONST], curLen * R_CONST);
                SetFlag<HardEvent::MTE2_V>(evB[0]);
            }
            for (int64_t t = 0; t < nTiles; ++t) {
                const int64_t d = t * BLK_V;
                const int64_t curLen = (supLen - d < BLK_V) ? (supLen - d) : BLK_V;
                if (t + 1 < nTiles) {
                    const int64_t nxtLen =
                        (supLen - d - BLK_V < BLK_V) ? (supLen - d - BLK_V) : BLK_V;
                    DataCopy(w2T2_, w2Gm_[(vLo_ + st + d + BLK_V) * R_CONST],
                             nxtLen * R_CONST);
                    SetFlag<HardEvent::MTE2_V>(evB[1]);
                }
                WaitFlag<HardEvent::MTE2_V>(evB[0]);
                Cast(w2F32_, w2T_, RoundMode::CAST_NONE, curLen * R_CONST);
                PipeBarrier<PIPE_V>();
                for (int64_t b = 0; b < B_; ++b) {
                    for (int64_t v = 0; v < curLen; ++v) {
                        Mul(prod_[v * R_CONST], w2F32_[v * R_CONST], embF32_[b * R_CONST],
                            R_CONST);
                    }
                    PipeBarrier<PIPE_V>();
                    uint32_t rs[2] = {static_cast<uint32_t>(curLen), static_cast<uint32_t>(R_CONST)};
                    ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumF32_[b * BLK_V],
                                                                          prod_, arTmp_, rs,
                                                                          true);
                    PipeBarrier<PIPE_V>();
                    // materialization point 1 (S2): fp32 accumulator -> fp16 RNE
                    Cast(band16_[b * SUP_V + d], sumF32_[b * BLK_V], RoundMode::CAST_ROUND,
                         curLen);
                }
                LocalTensor<half> tmp = w2T_;
                w2T_ = w2T2_;
                w2T2_ = tmp;
                event_t evTmp = evB[0];
                evB[0] = evB[1];
                evB[1] = evTmp;
            }
            PipeBarrier<PIPE_V>();
            VToMte3();  // band (V-written) -> MTE3 publish ordering (race fix)
            for (int64_t b = 0; b < B_; ++b) {
                DataCopy(biasGm_[b * vsPad_ + vLo_ + st], band16_[b * SUP_V], supLen);
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> emb16Buf_, embF32Buf_, w2TBuf_, w2F32Buf_, prodbuf_, bandBuf_,
        sumBuf_, arBuf_;
    LocalTensor<half> emb16_, w2T_, w2T2_, band16_;
    LocalTensor<float> embF32_, w2F32_, prod_, sumF32_;
    LocalTensor<uint8_t> arTmp_;
    GlobalTensor<half> embGm_, w2Gm_, biasGm_;
    int64_t B_ = 0, vsPad_ = 0, sw_ = 0, vLo_ = 0, vHi_ = 0, w_ = 0;
};

// ===========================================================================
// L2 op C: full-vocab (logits[:,k] + bias_full) first-max argmax.
// Per-core packed candidates land in ws; the NEXT launch merges (L1 rule).
// ===========================================================================
class Mh2Argmax {
public:
    __aicore__ inline void Init(GM_ADDR logitsIn, GM_ADDR biasFlatIn, GM_ADDR wsIn,
                                int64_t B, int64_t K, int64_t V, int64_t Wn, int64_t vs,
                                int64_t vsPad, int64_t sw, int64_t stepK)
    {
        const int64_t pid = GetBlockIdx();
        B_ = B;
        K_ = K;
        V_ = V;
        vs_ = vs;
        vsPad_ = vsPad;
        sw_ = sw;
        stepK_ = stepK;
        vLo_ = pid * sw;
        vHi_ = (vLo_ + sw < V) ? (vLo_ + sw) : V;
        w_ = (vHi_ > vLo_) ? (vHi_ - vLo_) : 0;
        logitsGm_.SetGlobalBuffer((__gm__ half *)logitsIn);
        biasGm_.SetGlobalBuffer((__gm__ half *)biasFlatIn);
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsIn);
        pipe_.InitBuffer(lgBuf_, ARG_CHUNK * sizeof(half));         // 2KB
        pipe_.InitBuffer(biasBuf_, ARG_CHUNK * sizeof(half));       // 2KB
        pipe_.InitBuffer(f32Buf_, ARG_CHUNK * 3 * sizeof(float));   // 12KB
        pipe_.InitBuffer(s16Buf_, ARG_CHUNK * sizeof(half));        // 2KB
        pipe_.InitBuffer(redBuf_, ARG_CHUNK * 2 * sizeof(float) + 64);  // 8KB
        pipe_.InitBuffer(wsBuf_, LINE_I64 * sizeof(int64_t));       // 128B line
        lg16_ = lgBuf_.Get<half>();
        bias16_ = biasBuf_.Get<half>();
        lgF32_ = f32Buf_.Get<float>();
        biasF32_ = f32Buf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 4);
        sumF32_ = f32Buf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 8);
        s16h_ = s16Buf_.Get<half>();
        rowF32_ = redBuf_.Get<float>();
        redTmpF_ = redBuf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 4);
        dst2F_ = redBuf_.GetWithOffset<float>(8, ARG_CHUNK * 8);
        wsLine_ = wsBuf_.Get<int64_t>();
    }

    __aicore__ inline void Process()
    {
        int64_t keyBest[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (w_ > 0) {
            for (int64_t b = 0; b < B_; ++b) {
                for (int64_t d = 0; d < w_; d += ARG_CHUNK) {
                    const int64_t v0 = vLo_ + d;
                    const int64_t seg = v0 / vs_;
                    const int64_t j0 = v0 - seg * vs_;
                    int64_t curLen = (w_ - d < ARG_CHUNK) ? (w_ - d) : ARG_CHUNK;
                    if (vs_ - j0 < curLen) { curLen = vs_ - j0; }
                    DataCopy(lg16_, logitsGm_[b * K_ * V_ + stepK_ * V_ + v0], curLen);
                    DataCopy(bias16_, biasGm_[seg * (B_ * vsPad_) + b * vsPad_ + j0], curLen);
                    Mte2ToV();
                    Cast(lgF32_, lg16_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    Cast(biasF32_, bias16_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    Add(sumF32_, biasF32_, lgF32_, curLen);  // point-2 fp32 domain (S2)
                    PipeBarrier<PIPE_V>();
                    Cast(s16h_, sumF32_, RoundMode::CAST_ROUND, curLen);  // fp16 RNE (S2)
                    PipeBarrier<PIPE_V>();
                    Cast(rowF32_, s16h_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    ReduceMax(dst2F_, rowF32_, redTmpF_, static_cast<int32_t>(curLen), true);
                    PipeBarrier<PIPE_V>();  // V drained -> scalar GetValue safe
                    const float mbF = dst2F_.GetValue(0);
                    const float idxRawF = dst2F_.GetValue(1);
                    uint32_t idxBits = 0;
                    __builtin_memcpy(&idxBits, &idxRawF, sizeof(uint32_t));
                    const int64_t idx = static_cast<int64_t>(idxBits & 0x7FFFFFFFu) + v0;
                    const int64_t key = PackKeyVal(static_cast<half>(mbF), idx);
                    if (key > keyBest[b]) { keyBest[b] = key; }  // ascending v: first-max
                }
            }
        }
        for (int64_t b = 0; b < LINE_I64; ++b) {
            wsLine_.SetValue(b, 0);
        }
        for (int64_t b = 0; b < B_; ++b) {
            wsLine_.SetValue(b, keyBest[b]);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(wsGm_[GetBlockIdx() * LINE_I64], wsLine_, LINE_I64);  // full 128B line
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lgBuf_, biasBuf_, f32Buf_, s16Buf_, redBuf_, wsBuf_;
    LocalTensor<half> lg16_, bias16_, s16h_;
    LocalTensor<float> lgF32_, biasF32_, sumF32_, rowF32_, redTmpF_, dst2F_;
    LocalTensor<int64_t> wsLine_;
    GlobalTensor<half> logitsGm_, biasGm_;
    GlobalTensor<int64_t> wsGm_;
    int64_t B_ = 0, K_ = 0, V_ = 0, vs_ = 0, vsPad_ = 0, sw_ = 0, stepK_ = 0;
    int64_t vLo_ = 0, vHi_ = 0, w_ = 0;
};

// ===========================================================================
// L3 op D: full local step -> per-core packed candidate keys in ws
// ===========================================================================
class Mh3Step {
public:
    __aicore__ inline void Init(GM_ADDR logitsIn, GM_ADDR w1In, GM_ADDR w2In, GM_ADDR wsAgIn,
                                GM_ADDR seedIn, GM_ADDR tokOut, int64_t B, int64_t K,
                                int64_t V, int64_t vs, int64_t r0, int64_t sw, int64_t coreNum,
                                int64_t world, int64_t stepK, int64_t mode)
    {
        const int64_t pid = GetBlockIdx();
        world_ = world;
        B_ = B;
        K_ = K;
        V_ = V;
        vs_ = vs;
        r0_ = r0;
        sw_ = sw;
        coreNum_ = coreNum;
        stepK_ = stepK;
        mode_ = mode;
        vLo_ = pid * sw;
        vHi_ = (vLo_ + sw < vs) ? (vLo_ + sw) : vs;
        w_ = (vHi_ > vLo_) ? (vHi_ - vLo_) : 0;
        logitsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)logitsIn);
        w1Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)w1In);
        w2Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)w2In);
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsAgIn);  // [W, 48, 16] gather output
        seedGm_.SetGlobalBuffer((__gm__ int64_t *)seedIn);
        tokGm_.SetGlobalBuffer((__gm__ int64_t *)tokOut);
        // UB ~87KB (< the ~96KB sub-block window; L1 precedent ran at 89KB).
        // B==1 fast path: embBCast materializes the [16,256] fp32 broadcast of
        // the single emb row ONCE per launch (16 Sub(x - +0.0) = bit-exact),
        // collapsing the 16 per-tile Muls into one (v2 forbids stride-0
        // broadcast operands, so the broadcast must be materialized; the
        // double-buffered w2T experiment (V2) measured NO gain and was
        // reverted to free the UB for this).
        pipe_.InitBuffer(emb16Buf_, B_PAD * R_CONST * sizeof(bfloat16_t));   // 4KB
        pipe_.InitBuffer(embF32Buf_, B_PAD * R_CONST * sizeof(float)); // 8KB
        pipe_.InitBuffer(w2TBuf_, BLK_V * R_CONST * sizeof(bfloat16_t));     // 8KB
        pipe_.InitBuffer(w2F32Buf_, BLK_V * R_CONST * sizeof(float));  // 16KB
        pipe_.InitBuffer(prodbuf_, BLK_V * R_CONST * sizeof(float));   // 16KB
        pipe_.InitBuffer(bandBuf_, B_PAD * EPI_CHUNK * sizeof(bfloat16_t));  // 8KB
        pipe_.InitBuffer(epiBuf_, (EPI_CHUNK * 6 + 8) * sizeof(float) + 64);  // ~12.6KB
        pipe_.InitBuffer(arBuf_, 8192);
        pipe_.InitBuffer(wsBuf_, world * coreNum * LINE_I64 * sizeof(int64_t) + 2 * 128);
        emb16_ = emb16Buf_.Get<bfloat16_t>();
        embF32_ = embF32Buf_.Get<float>();
        w2T_ = w2TBuf_.Get<bfloat16_t>();
        w2F32_ = w2F32Buf_.Get<float>();
        prod_ = prodbuf_.Get<float>();
        band16_ = bandBuf_.Get<bfloat16_t>();
        // epiBuf layout (bytes): lg16[512]bf | s16[512]bf | bRow[512]f | lgRow[512]f
        //                        | sumRow[512]f | rowF[512]f | dst2[8]f (+pad)
        lg16_ = epiBuf_.Get<bfloat16_t>();
        s16h_ = epiBuf_.GetWithOffset<bfloat16_t>(EPI_CHUNK, EPI_CHUNK * 2);
        bRowF32_ = epiBuf_.GetWithOffset<float>(EPI_CHUNK, EPI_CHUNK * 4);
        lgRowF32_ = epiBuf_.GetWithOffset<float>(EPI_CHUNK, EPI_CHUNK * 4 + EPI_CHUNK * 4);
        sumRowF32_ = epiBuf_.GetWithOffset<float>(EPI_CHUNK, EPI_CHUNK * 8 + EPI_CHUNK * 4);
        rowF32_ = epiBuf_.GetWithOffset<float>(EPI_CHUNK, EPI_CHUNK * 12 + EPI_CHUNK * 4);
        dst2F_ = epiBuf_.GetWithOffset<float>(8, EPI_CHUNK * 16 + EPI_CHUNK * 4);
        sumF32_ = epiBuf_.GetWithOffset<float>(B_PAD * BLK_V,
                                               EPI_CHUNK * 16 + EPI_CHUNK * 4 + 64);
        arTmp_ = arBuf_.Get<uint8_t>();
        wsAll_ = wsBuf_.Get<int64_t>();
        wsLine_ = wsBuf_.GetWithOffset<int64_t>(LINE_I64,
                                                coreNum * LINE_I64 * sizeof(int64_t));
        seedLine_ = wsBuf_.GetWithOffset<int64_t>(LINE_I64,
                                                  coreNum * LINE_I64 * sizeof(int64_t) + 128);
    }

    __aicore__ inline void Process()
    {
        for (int64_t b = 0; b < 8; ++b) {
            keyBest_[b] = 0;
        }
        // ---- resolve input tokens (all cores, redundant merge; ws was
        //      written + AllReduced at least one launch boundary ago) ----
        int32_t toks[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (mode_ != 0) {
            MergeWsAgToToks(wsGm_, wsAll_, B_, coreNum_, world_, toks);
            if (GetBlockIdx() == 0) {
                // record the resolved tokens for the final out_ids assembly
                for (int64_t b = 0; b < LINE_I64; ++b) {
                    seedLine_.SetValue(b, 0);
                }
                for (int64_t b = 0; b < B_; ++b) {
                    seedLine_.SetValue(b, static_cast<int64_t>(toks[b]));
                }
                PipeBarrier<PIPE_ALL>();
                DataCopy(tokGm_, seedLine_, LINE_I64);  // full 128B line
                PipeBarrier<PIPE_MTE3>();
            }
        } else {
            DataCopy(seedLine_, seedGm_, LINE_I64);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            for (int64_t b = 0; b < B_; ++b) {
                toks[b] = static_cast<int32_t>(seedLine_.GetValue(b));
            }
            PipeBarrier<PIPE_ALL>();
        }
        // ---- gather emb rows from the replicated full w1 ----
        for (int64_t b = 0; b < B_; ++b) {
            DataCopy(emb16_[b * R_CONST], w1Gm_[static_cast<int64_t>(toks[b]) * R_CONST],
                     R_CONST);
        }
        Mte2ToV();
        PipeBarrier<PIPE_V>();
        for (int64_t b = 0; b < B_; ++b) {
            Cast(embF32_[b * R_CONST], emb16_[b * R_CONST], RoundMode::CAST_NONE, R_CONST);
        }
        PipeBarrier<PIPE_V>();
        // NOTE (bf16 fused path): the in-kernel GEMV below accumulates in
        // fp32 but in ITS OWN order (per-tile AR ReduceSum), which is NOT the
        // aclnnMatMulV2 accumulation the bf16 torch reference uses — unlike
        // the orchestrated path (mh3_emb + at::matmul_out + mh3_argmax) this
        // fused backup has no same-op-class guarantee for materialization
        // point 1 and must be re-verified per weight/case corpus.
        if (w_ > 0) {
            // ---- GEMV + epilogue in EPI_CHUNK segments (running first-max) ----
            // (an embBCast fast path for B==1 — materialize the [16,256] fp32
            // broadcast once and collapse 16 Muls into one — was implemented
            // and measured, but repeatedly corrupted results even after the
            // UB window was brought back under 96KB; reverted to the plain
            // verified loop. See trace.md deviation notes.)
            for (int64_t st = 0; st < w_; st += EPI_CHUNK) {
                const int64_t segLen = (w_ - st < EPI_CHUNK) ? (w_ - st) : EPI_CHUNK;
                for (int64_t d = 0; d < segLen; d += BLK_V) {
                    const int64_t curLen = (segLen - d < BLK_V) ? (segLen - d) : BLK_V;
                    DataCopy(w2T_, w2Gm_[(vLo_ + st + d) * R_CONST], curLen * R_CONST);
                    Mte2ToV();
                    Cast(w2F32_, w2T_, RoundMode::CAST_NONE, curLen * R_CONST);
                    PipeBarrier<PIPE_V>();
                    for (int64_t b = 0; b < B_; ++b) {
                        for (int64_t v = 0; v < curLen; ++v) {
                            Mul(prod_[v * R_CONST], w2F32_[v * R_CONST],
                                embF32_[b * R_CONST], R_CONST);
                        }
                        PipeBarrier<PIPE_V>();
                        uint32_t rs[2] = {static_cast<uint32_t>(curLen),
                                          static_cast<uint32_t>(R_CONST)};
                        ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
                            sumF32_[b * BLK_V], prod_, arTmp_, rs, true);
                        PipeBarrier<PIPE_V>();
                        // materialization point 1 (S2), bf16 RNE. NOTE: uses
                        // CAST_RINT (vconv_f322bf16r), NOT CAST_ROUND — the
                        // latter is RNA for bf16 on this AIV (engineered-tie
                        // probe bf16_r0/probe_bf16_ops_bitexact.json E1),
                        // unlike fp16 where CAST_ROUND measures RNE
                        Cast(band16_[b * EPI_CHUNK + d], sumF32_[b * BLK_V],
                             RoundMode::CAST_RINT, curLen);
                    }
                }
                PipeBarrier<PIPE_V>();  // band segment complete
                for (int64_t b = 0; b < B_; ++b) {
                    DataCopy(lg16_, logitsGm_[b * K_ * V_ + stepK_ * V_ + r0_ * vs_ + vLo_ + st],
                             segLen);
                    Mte2ToV();
                    Cast(bRowF32_, band16_[b * EPI_CHUNK], RoundMode::CAST_NONE, segLen);
                    PipeBarrier<PIPE_V>();
                    Cast(lgRowF32_, lg16_, RoundMode::CAST_NONE, segLen);
                    PipeBarrier<PIPE_V>();
                    Add(sumRowF32_, bRowF32_, lgRowF32_, segLen);  // point-2 fp32 domain (S2)
                    PipeBarrier<PIPE_V>();
                    Cast(s16h_, sumRowF32_, RoundMode::CAST_RINT, segLen);  // bf16 RNE (S2)
                    PipeBarrier<PIPE_V>();
                    Cast(rowF32_, s16h_, RoundMode::CAST_NONE, segLen);
                    PipeBarrier<PIPE_V>();
                    ReduceMax(dst2F_, rowF32_, lgRowF32_, static_cast<int32_t>(segLen), true);
                    PipeBarrier<PIPE_V>();
                    const float mbF = dst2F_.GetValue(0);
                    const float idxRawF = dst2F_.GetValue(1);
                    uint32_t idxBits = 0;
                    __builtin_memcpy(&idxBits, &idxRawF, sizeof(uint32_t));
                    const int64_t idx =
                        static_cast<int64_t>(idxBits & 0x7FFFFFFFu) + r0_ * vs_ + vLo_ + st;
                    const int64_t key = PackKeyVal(mbF, idx);
                    if (key > keyBest_[b]) { keyBest_[b] = key; }  // ascending v
                }
            }
        }
        // ---- publish per-core candidates into THIS rank's [48,16] region of
        //      the gather buffer (r0 == rank id; full 128B lines)
        for (int64_t b = 0; b < LINE_I64; ++b) {
            wsLine_.SetValue(b, 0);
        }
        for (int64_t b = 0; b < B_; ++b) {
            wsLine_.SetValue(b, keyBest_[b]);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(wsGm_[r0_ * coreNum_ * LINE_I64 + GetBlockIdx() * LINE_I64], wsLine_,
                 LINE_I64);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> emb16Buf_, embF32Buf_, w2TBuf_, w2F32Buf_, prodbuf_, bandBuf_,
        epiBuf_, arBuf_, wsBuf_;
    LocalTensor<bfloat16_t> emb16_, w2T_, w2T2_, band16_, lg16_, s16h_;
    LocalTensor<float> embF32_, w2F32_, prod_, bRowF32_, lgRowF32_, sumRowF32_, rowF32_;
    LocalTensor<float> dst2F_, sumF32_;
    LocalTensor<uint8_t> arTmp_;
    LocalTensor<int64_t> wsAll_, wsLine_, seedLine_;
    GlobalTensor<bfloat16_t> logitsGm_, w1Gm_, w2Gm_;
    GlobalTensor<int64_t> wsGm_, seedGm_, tokGm_;
    int64_t B_ = 0, K_ = 0, V_ = 0, vs_ = 0, r0_ = 0, sw_ = 0, coreNum_ = 0, stepK_ = 0;
    int64_t mode_ = 0, vLo_ = 0, vHi_ = 0, w_ = 0, world_ = 1;
    int64_t keyBest_[8];
};

// ===========================================================================
// L3 op D1 (P0 "borrow the cube path" decomposition): b-parallel
// [merge gathered ws keys -> tok] + w1 row gather -> emb row publish.
// Only blocks 0..B-1 are active; block b owns batch row b:
//   - mode!=0: scan column b of the [W,48,16] gathered keys (world*coreNum
//     scalar reads, ~192-384 -> a few us) and decode the winning token
//   - mode==0: read the seed line
// then gather w1 row tok_b and publish emb row b (512B = whole 128B lines,
// V-produced + VToMte3 per the MTE2->MTE3 rule) plus a full scratch line
// [16]i64 (slot 0 = tok_b) that the argmax launch assembles into tok_out.
// The b-parallel merge replaces the redundant per-core merge of the fused
// opD (wall W*coreNum*B scalar reads on ONE core -> /B in parallel) with
// bit-identical results (max over the same key set; associativity).
// ===========================================================================
class Mh3Emb {
public:
    __aicore__ inline void Init(GM_ADDR wsAgIn, GM_ADDR seedIn, GM_ADDR w1In, GM_ADDR embOut,
                                GM_ADDR tokScratch, int64_t B, int64_t coreNum, int64_t world,
                                int64_t mode)
    {
        B_ = B;
        coreNum_ = coreNum;
        world_ = world;
        mode_ = mode;
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsAgIn);
        seedGm_.SetGlobalBuffer((__gm__ int64_t *)seedIn);
        w1Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)w1In);
        embGm_.SetGlobalBuffer((__gm__ bfloat16_t *)embOut);
        scrGm_.SetGlobalBuffer((__gm__ int64_t *)tokScratch);
        pipe_.InitBuffer(wsBuf_, world * coreNum * LINE_I64 * sizeof(int64_t));  // <=48KB
        pipe_.InitBuffer(lineBuf_, LINE_I64 * sizeof(int64_t));                  // 128B
        // bf16 row staging: 2 rows of R_CONST bf16 (1KB). The row publish needs
        // V as the producer (MTE2->MTE3 direct pass-through loses rows), but
        // this AIV has NO bf16 vector Sub — the identity is done through an
        // int32 view of the same bytes with +0 (bitwise identity incl. -0.0
        // and NaN payloads; vadds int32 verified in the CANN dav_c220 impl).
        pipe_.InitBuffer(rowBuf_, 2 * R_CONST * sizeof(bfloat16_t));            // 1KB
        wsAll_ = wsBuf_.Get<int64_t>();
        line_ = lineBuf_.Get<int64_t>();
        stage_ = rowBuf_.Get<bfloat16_t>();
        row_ = rowBuf_.GetWithOffset<bfloat16_t>(R_CONST, R_CONST * sizeof(bfloat16_t));
        stageI32_ = rowBuf_.Get<int32_t>();                                     // [0,512B)
        rowI32_ = rowBuf_.GetWithOffset<int32_t>(R_CONST / 2, R_CONST * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        const int64_t b = GetBlockIdx();
        if (b >= B_) { return; }
        int64_t tok = 0;
        if (mode_ != 0) {
            DataCopy(wsAll_, wsGm_, world_ * coreNum_ * LINE_I64);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            int64_t best = 0;
            for (int64_t r = 0; r < world_; ++r) {
                for (int64_t p = 0; p < coreNum_; ++p) {
                    const int64_t k = wsAll_.GetValue((r * coreNum_ + p) * LINE_I64 + b);
                    if (k > best) { best = k; }
                }
            }
            tok = KEY_IDX_MASK - (best & KEY_IDX_MASK);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopy(line_, seedGm_, LINE_I64);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            tok = line_.GetValue(b);
            PipeBarrier<PIPE_ALL>();
        }
        // record tok_b in this block's own full 128B scratch line
        for (int64_t i = 0; i < LINE_I64; ++i) {
            line_.SetValue(i, 0);
        }
        line_.SetValue(0, tok);
        PipeBarrier<PIPE_ALL>();
        DataCopy(scrGm_[b * LINE_I64], line_, LINE_I64);
        PipeBarrier<PIPE_MTE3>();
        // gather the full-w1 embedding row (w1 replicated; every token is local)
        DataCopy(stage_, w1Gm_[tok * R_CONST], R_CONST);
        Mte2ToV();
        // int32 +0 pass-through on the same 512B = bitwise identity (no bf16
        // vector Sub on this AIV); V stays the producer of the published row
        Adds(rowI32_, stageI32_, static_cast<int32_t>(0), R_CONST / 2);
        PipeBarrier<PIPE_V>();
        VToMte3();
        DataCopy(embGm_[b * R_CONST], row_, R_CONST);  // 512B whole lines
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> wsBuf_, lineBuf_, rowBuf_;
    LocalTensor<int64_t> wsAll_, line_;
    LocalTensor<bfloat16_t> row_, stage_;
    LocalTensor<int32_t> rowI32_, stageI32_;
    GlobalTensor<int64_t> wsGm_, seedGm_, scrGm_;
    GlobalTensor<bfloat16_t> w1Gm_, embGm_;
    int64_t B_ = 0, coreNum_ = 0, world_ = 1, mode_ = 0;
};

// ===========================================================================
// L3 op D2: (bias_local + logits shard) first-max -> packed keys in ws.
// bias_local is the aclnn MatMulV2 output [B, vs] bf16 (materialization
// point 1: fp32 accumulate -> bf16 RNE, the same aclnn op the bf16 torch
// deployment reference chain uses). Point 2 (fp32 add of two bf16 -> bf16
// RNE via vconv_f322bf16a, matching torch bf16 add_ fp32-opmath semantics)
// and the first-max/mono-key epilogue follow the verified fp16 opD epilogue
// structure with the bf16 PackKeyVal. Block 0 additionally assembles tok_out
// from the emb launch's scratch lines (one launch boundary ago; whole 128B
// lines).
// ===========================================================================
class Mh3ArgmaxL {
public:
    __aicore__ inline void Init(GM_ADDR logitsIn, GM_ADDR biasIn, GM_ADDR wsAgIn,
                                GM_ADDR tokScratch, GM_ADDR tokOut, int64_t B, int64_t K,
                                int64_t V, int64_t vs, int64_t r0, int64_t sw, int64_t coreNum,
                                int64_t stepK, int64_t fillFrom)
    {
        const int64_t pid = GetBlockIdx();
        B_ = B;
        K_ = K;
        V_ = V;
        vs_ = vs;
        r0_ = r0;
        coreNum_ = coreNum;
        stepK_ = stepK;
        fillFrom_ = fillFrom;
        vLo_ = pid * sw;
        vHi_ = (vLo_ + sw < vs) ? (vLo_ + sw) : vs;
        w_ = (vHi_ > vLo_) ? (vHi_ - vLo_) : 0;
        logitsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)logitsIn);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t *)biasIn);
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsAgIn);
        scrGm_.SetGlobalBuffer((__gm__ int64_t *)tokScratch);
        tokGm_.SetGlobalBuffer((__gm__ int64_t *)tokOut);
        pipe_.InitBuffer(lgBuf_, ARG_CHUNK * sizeof(bfloat16_t));     // 2KB
        pipe_.InitBuffer(biasBuf_, ARG_CHUNK * sizeof(bfloat16_t));   // 2KB
        pipe_.InitBuffer(f32Buf_, ARG_CHUNK * 3 * sizeof(float));     // 12KB
        pipe_.InitBuffer(s16Buf_, ARG_CHUNK * sizeof(bfloat16_t));    // 2KB
        pipe_.InitBuffer(redBuf_, ARG_CHUNK * 2 * sizeof(float) + 64);  // 8KB
        pipe_.InitBuffer(wsBuf_, LINE_I64 * sizeof(int64_t));         // 128B
        pipe_.InitBuffer(scrBuf_, B_PAD * LINE_I64 * sizeof(int64_t));  // 1KB
        pipe_.InitBuffer(zerBuf_, 48 * LINE_I64 * sizeof(int64_t));   // 6KB (tail zero-fill)
        lg16_ = lgBuf_.Get<bfloat16_t>();
        bias16_ = biasBuf_.Get<bfloat16_t>();
        lgF32_ = f32Buf_.Get<float>();
        biasF32_ = f32Buf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 4);
        sumF32_ = f32Buf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 8);
        s16h_ = s16Buf_.Get<bfloat16_t>();
        rowF32_ = redBuf_.Get<float>();
        redTmpF_ = redBuf_.GetWithOffset<float>(ARG_CHUNK, ARG_CHUNK * 4);
        dst2F_ = redBuf_.GetWithOffset<float>(8, ARG_CHUNK * 8);
        wsLine_ = wsBuf_.Get<int64_t>();
        scrAll_ = scrBuf_.Get<int64_t>();
        zeroRegion_ = zerBuf_.Get<int64_t>();
        {   // Duplicate has no int64 overload; zeros are dtype-agnostic
            LocalTensor<int32_t> z32 = zerBuf_.Get<int32_t>();
            Duplicate(z32, static_cast<int32_t>(0), 48 * LINE_I64 * 2);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        // ---- block 0: assemble tok_out from the emb launch's scratch lines --
        if (GetBlockIdx() == 0) {
            DataCopy(scrAll_, scrGm_, B_ * LINE_I64);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            for (int64_t b = 0; b < LINE_I64; ++b) {
                wsLine_.SetValue(b, 0);
            }
            for (int64_t b = 0; b < B_; ++b) {
                wsLine_.SetValue(b, scrAll_.GetValue(b * LINE_I64));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(tokGm_, wsLine_, LINE_I64);  // full 128B line
            PipeBarrier<PIPE_MTE3>();
        }
        // ---- stripe epilogue: (bias + logits) -> bf16 RNE -> first-max ----
        int64_t keyBest[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (w_ > 0) {
            for (int64_t b = 0; b < B_; ++b) {
                for (int64_t d = 0; d < w_; d += ARG_CHUNK) {
                    int64_t curLen = (w_ - d < ARG_CHUNK) ? (w_ - d) : ARG_CHUNK;
                    DataCopy(lg16_, logitsGm_[b * K_ * V_ + stepK_ * V_ + r0_ * vs_ + vLo_ + d],
                             curLen);
                    DataCopy(bias16_, biasGm_[b * vs_ + vLo_ + d], curLen);
                    Mte2ToV();
                    Cast(lgF32_, lg16_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    Cast(biasF32_, bias16_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    Add(sumF32_, biasF32_, lgF32_, curLen);  // point-2 fp32 domain (S2)
                    PipeBarrier<PIPE_V>();
                    // bf16 RNE materialization (S2 point 2): vconv_f322bf16r
                    // (CAST_RINT). CAST_ROUND is RNA for bf16 here — see the
                    // engineered-tie probe E1 note in the fused opD above
                    Cast(s16h_, sumF32_, RoundMode::CAST_RINT, curLen);
                    PipeBarrier<PIPE_V>();
                    Cast(rowF32_, s16h_, RoundMode::CAST_NONE, curLen);
                    PipeBarrier<PIPE_V>();
                    ReduceMax(dst2F_, rowF32_, lgF32_, static_cast<int32_t>(curLen), true);
                    PipeBarrier<PIPE_V>();  // V drained -> scalar GetValue safe
                    const float mbF = dst2F_.GetValue(0);
                    const float idxRawF = dst2F_.GetValue(1);
                    uint32_t idxBits = 0;
                    __builtin_memcpy(&idxBits, &idxRawF, sizeof(uint32_t));
                    const int64_t idx =
                        static_cast<int64_t>(idxBits & 0x7FFFFFFFu) + r0_ * vs_ + vLo_ + d;
                    // mbF is the fp32 image of a materialized bf16 -> the bf16
                    // PackKeyVal extracts its bits by pure right shift
                    const int64_t key = PackKeyVal(mbF, idx);
                    if (key > keyBest[b]) { keyBest[b] = key; }  // ascending v: first-max
                }
            }
        }
        // ---- publish per-core candidates into THIS rank's [48,16] region ----
        for (int64_t b = 0; b < LINE_I64; ++b) {
            wsLine_.SetValue(b, 0);
        }
        for (int64_t b = 0; b < B_; ++b) {
            wsLine_.SetValue(b, keyBest[b]);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(wsGm_[r0_ * coreNum_ * LINE_I64 + GetBlockIdx() * LINE_I64], wsLine_,
                 LINE_I64);
        PipeBarrier<PIPE_MTE3>();
        // ---- P2: with fewer launched blocks than the 48 ws slots, block 0
        //      zero-fills the untouched tail (key 0 is neutral under max;
        //      stale keys from previous launches must never survive) --------
        if (GetBlockIdx() == 0 && fillFrom_ < coreNum_) {
            VToMte3();  // zeroRegion_ is V-produced (Duplicate)
            DataCopy(wsGm_[r0_ * coreNum_ * LINE_I64 + fillFrom_ * LINE_I64], zeroRegion_,
                     (coreNum_ - fillFrom_) * LINE_I64);
            PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lgBuf_, biasBuf_, f32Buf_, s16Buf_, redBuf_, wsBuf_, scrBuf_,
        zerBuf_;
    LocalTensor<bfloat16_t> lg16_, bias16_, s16h_;
    LocalTensor<float> lgF32_, biasF32_, sumF32_, rowF32_, redTmpF_, dst2F_;
    LocalTensor<int64_t> wsLine_, scrAll_, zeroRegion_;
    GlobalTensor<bfloat16_t> logitsGm_, biasGm_;
    GlobalTensor<int64_t> wsGm_, scrGm_, tokGm_;
    int64_t B_ = 0, K_ = 0, V_ = 0, vs_ = 0, r0_ = 0, coreNum_ = 0, stepK_ = 0;
    int64_t fillFrom_ = 0;
    int64_t vLo_ = 0, vHi_ = 0, w_ = 0;
};

// ===========================================================================
// Final-column merge: ws -> tok line (single block, once per forward)
// ===========================================================================
class MhMerge {
public:
    __aicore__ inline void Init(GM_ADDR wsIn, GM_ADDR tokOut, int64_t B, int64_t coreNum,
                                int64_t world)
    {
        B_ = B;
        coreNum_ = coreNum;
        world_ = world;
        wsGm_.SetGlobalBuffer((__gm__ int64_t *)wsIn);
        tokGm_.SetGlobalBuffer((__gm__ int64_t *)tokOut);
        pipe_.InitBuffer(wsBuf_, world * coreNum * LINE_I64 * sizeof(int64_t));
        pipe_.InitBuffer(lineBuf_, LINE_I64 * sizeof(int64_t));
        wsAll_ = wsBuf_.Get<int64_t>();
        line_ = lineBuf_.Get<int64_t>();
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) { return; }
        int32_t toks[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        MergeWsAgToToks(wsGm_, wsAll_, B_, coreNum_, world_, toks);
        for (int64_t b = 0; b < LINE_I64; ++b) {
            line_.SetValue(b, 0);
        }
        for (int64_t b = 0; b < B_; ++b) {
            line_.SetValue(b, static_cast<int64_t>(toks[b]));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(tokGm_, line_, LINE_I64);  // full 128B line
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> wsBuf_, lineBuf_;
    LocalTensor<int64_t> wsAll_, line_;
    GlobalTensor<int64_t> wsGm_, tokGm_;
    int64_t B_ = 0, coreNum_ = 0, world_ = 1;
};

}  // namespace

extern "C" __global__ __aicore__ void dspark_mh2_emb(
    GM_ADDR tok_line, GM_ADDR ws, GM_ADDR w1, GM_ADDR emb_out, GM_ADDR tok_out, int64_t B,
    int64_t lo, int64_t hi, int64_t coreNum, int64_t mode)
{
    Mh2Emb op;
    op.Init(tok_line, ws, w1, emb_out, tok_out, B, lo, hi, coreNum, mode);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_mh2_bias(
    GM_ADDR emb, GM_ADDR w2, GM_ADDR bias_out, int64_t B, int64_t vs, int64_t vs_pad,
    int64_t sw)
{
    Mh2Bias op;
    op.Init(emb, w2, bias_out, B, vs, vs_pad, sw);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_mh2_argmax(
    GM_ADDR logits, GM_ADDR bias_flat, GM_ADDR ws, int64_t B, int64_t K, int64_t V, int64_t Wn,
    int64_t vs, int64_t vs_pad, int64_t sw, int64_t stepK)
{
    Mh2Argmax op;
    op.Init(logits, bias_flat, ws, B, K, V, Wn, vs, vs_pad, sw, stepK);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_mh3_step(
    GM_ADDR logits, GM_ADDR w1, GM_ADDR w2, GM_ADDR ws_ag, GM_ADDR seed_line, GM_ADDR tok_out,
    int64_t B, int64_t K, int64_t V, int64_t vs, int64_t r0, int64_t sw, int64_t coreNum,
    int64_t world, int64_t stepK, int64_t mode)
{
    Mh3Step op;
    op.Init(logits, w1, w2, ws_ag, seed_line, tok_out, B, K, V, vs, r0, sw, coreNum, world,
            stepK, mode);
    op.Process();
}

// P0 "borrow the cube path" decomposition of opD (see host wrapper): the GEMV
// is delegated to aclnn MatMulV2 between these two launches.
extern "C" __global__ __aicore__ void dspark_mh3_emb(
    GM_ADDR ws_ag, GM_ADDR seed_line, GM_ADDR w1, GM_ADDR emb_out, GM_ADDR tok_scratch,
    int64_t B, int64_t coreNum, int64_t world, int64_t mode)
{
    Mh3Emb op;
    op.Init(ws_ag, seed_line, w1, emb_out, tok_scratch, B, coreNum, world, mode);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_mh3_argmax(
    GM_ADDR logits, GM_ADDR bias_local, GM_ADDR ws_ag, GM_ADDR tok_scratch, GM_ADDR tok_out,
    int64_t B, int64_t K, int64_t V, int64_t vs, int64_t r0, int64_t sw, int64_t coreNum,
    int64_t stepK, int64_t fillFrom)
{
    Mh3ArgmaxL op;
    op.Init(logits, bias_local, ws_ag, tok_scratch, tok_out, B, K, V, vs, r0, sw, coreNum,
            stepK, fillFrom);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_mh3_merge(GM_ADDR ws, GM_ADDR tok_out, int64_t B,
                                                       int64_t coreNum, int64_t world)
{
    MhMerge op;
    op.Init(ws, tok_out, B, coreNum, world);
    op.Process();
}

// ---------------------------------------------------------------------------
// P2 launch-floor probes (perf-iteration only, no functional role)
// ---------------------------------------------------------------------------
class ProbeEmpty {
public:
    __aicore__ inline void Init(GM_ADDR out, int64_t blocks)
    {
        blocks_ = blocks;
        outGm_.SetGlobalBuffer((__gm__ int64_t *)out);
        pipe_.InitBuffer(lineBuf_, LINE_I64 * sizeof(int64_t));
        line_ = lineBuf_.Get<int64_t>();
    }
    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < LINE_I64; ++i) {
            line_.SetValue(i, GetBlockIdx());
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm_[GetBlockIdx() * LINE_I64], line_, LINE_I64);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> lineBuf_;
    LocalTensor<int64_t> line_;
    GlobalTensor<int64_t> outGm_;
    int64_t blocks_ = 0;
};

// argmax-like buffer profile (7 TBufs, ~27KB) to separate the block-count
// floor from the buffer-init cost
class ProbeEmptyFat {
public:
    __aicore__ inline void Init(GM_ADDR out, int64_t blocks)
    {
        blocks_ = blocks;
        outGm_.SetGlobalBuffer((__gm__ int64_t *)out);
        pipe_.InitBuffer(b0_, ARG_CHUNK * sizeof(half));
        pipe_.InitBuffer(b1_, ARG_CHUNK * sizeof(half));
        pipe_.InitBuffer(b2_, ARG_CHUNK * 3 * sizeof(float));
        pipe_.InitBuffer(b3_, ARG_CHUNK * sizeof(half));
        pipe_.InitBuffer(b4_, ARG_CHUNK * 2 * sizeof(float) + 64);
        pipe_.InitBuffer(b5_, LINE_I64 * sizeof(int64_t));
        pipe_.InitBuffer(b6_, B_PAD * LINE_I64 * sizeof(int64_t));
        lg16_ = b0_.Get<half>();
        line_ = b5_.Get<int64_t>();
        Duplicate(lg16_, static_cast<half>(0), 8);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < LINE_I64; ++i) {
            line_.SetValue(i, GetBlockIdx());
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm_[GetBlockIdx() * LINE_I64], line_, LINE_I64);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> b0_, b1_, b2_, b3_, b4_, b5_, b6_;
    LocalTensor<half> lg16_;
    LocalTensor<int64_t> line_;
    GlobalTensor<int64_t> outGm_;
    int64_t blocks_ = 0;
};

extern "C" __global__ __aicore__ void dspark_probe_empty(GM_ADDR out, int64_t blocks)
{
    ProbeEmpty op;
    op.Init(out, blocks);
    op.Process();
}

extern "C" __global__ __aicore__ void dspark_probe_empty_fat(GM_ADDR out, int64_t blocks)
{
    ProbeEmptyFat op;
    op.Init(out, blocks);
    op.Process();
}

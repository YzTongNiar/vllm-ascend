/*
 * dspark_markov_head — DSpark sequential Markov chain fused kernel (AIV only).
 *
 * Semantics (golden reference, handover doc section 3.1):
 *   for k in 0..K-1:
 *     emb[b,:]  = w1[tok[b], :]                       # [B, 256] row gather
 *     bias      = fp32 GEMV(emb, w2^T) over [0, Vd)    # fp32 accumulate (S3)
 *     bias16    = fp16(bias)                           # materialization point 1 (S2)
 *     s16       = fp16(fp32(bias16) + fp32(logits))    # materialization point 2 (S2)
 *     tok'      = argmax_v(s16)                        # first-max tie-break (S4)
 *   out_ids    = [seed, tok_1 .. tok_K] int64          (S6)
 *
 * Structure (design/tile_level/dspark_markov_head.py + PERF_DESIGN.md):
 *   - single kernel launch, blockDim = number of AIV vector cores (D1)
 *   - K sequential steps inside the kernel, one SyncAll per step (user req,
 *     handover 6.2.4; the TileLang design layer expresses the same steps with
 *     launch boundaries because it has no cross-block barrier primitive)
 *   - two-region split (D2/D3): region1 [0, Vd) GEMV stripe, region2 [Vd, V)
 *     pure argmax stripe (bias == 0 for v >= Vd, red line S5); each region
 *     balanced across all cores with 16-element aligned stripe widths
 *   - zero intermediate GM materialization (D7); w2 read through L2 (D6)
 *   - argmax algebra (D5): per-stripe band argmax (first-max inside the band,
 *     band covers the whole stripe in ascending v) -> per-core candidate ->
 *     lexicographic (val, idx) cross-core merge, independent of core order
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int64_t BLK_V = 16;     // region-1 GEMV tile width (v columns)
constexpr int64_t BLK_V2 = 512;    // region-2 argmax tile width
constexpr int64_t CHUNK2 = 1024;  // region-2 chunk width ([1024]f spans idxBaseF_|infF_ slots)
constexpr int64_t B_PAD = 8;      // B <= 7 padded to 8 (fp32 row = 32B)
constexpr int64_t R_CONST = 256;  // markov rank (host asserts w1.size(1) == 256)
constexpr uint64_t MASK_ALL = 0xFFFFFFFFFFFFFFFFULL;
#ifndef DSPARK_CUT
#define DSPARK_CUT 99
#ifndef DSPARK_SKIP
#define DSPARK_SKIP 0
#endif
#endif

class DsparkMarkovChain {
public:
    __aicore__ inline DsparkMarkovChain() {}

    __aicore__ inline void Init(GM_ADDR logitsIn, GM_ADDR idsIn, GM_ADDR seed32In, GM_ADDR w1In,
                                GM_ADDR w2In, GM_ADDR outIn, GM_ADDR wsValIn, GM_ADDR wsIdxIn,
                                GM_ADDR scratchIn, GM_ADDR idxBaseIn, int64_t B, int64_t K,
                                int64_t V, int64_t Vd, int64_t sw1, int64_t sw2, int64_t coreNum,
                                int64_t stepK, int64_t isFinal)
    {
        pid_ = GetBlockIdx();
        B_ = B;
        K_ = K;
        V_ = V;
        Vd_ = Vd;
        sw1_ = sw1;
        sw2_ = sw2;
        coreNum_ = coreNum;
        stepK_ = stepK;
        isFinal_ = isFinal;
        vLo1_ = pid_ * sw1_;
        vHi1_ = (vLo1_ + sw1_ < Vd_) ? (vLo1_ + sw1_) : Vd_;
        w1_ = (vHi1_ > vLo1_) ? (vHi1_ - vLo1_) : 0;
        vLo2_ = Vd_ + pid_ * sw2_;
        vHi2_ = (vLo2_ + sw2_ < V_) ? (vLo2_ + sw2_) : V_;
        if (vLo2_ >= V_) { vLo2_ = V_; vHi2_ = V_; }
        w2_ = (vHi2_ > vLo2_) ? (vHi2_ - vLo2_) : 0;

        logitsGm_.SetGlobalBuffer((__gm__ half *)logitsIn);
        idsGm_.SetGlobalBuffer((__gm__ int64_t *)idsIn);
        seed32Gm_.SetGlobalBuffer((__gm__ int32_t *)seed32In);
        w1Gm_.SetGlobalBuffer((__gm__ half *)w1In);
        w2Gm_.SetGlobalBuffer((__gm__ half *)w2In);
        outGm_.SetGlobalBuffer((__gm__ int64_t *)outIn);
        wsValGm_.SetGlobalBuffer((__gm__ float *)wsValIn);
        wsIdxGm_.SetGlobalBuffer((__gm__ float *)wsIdxIn);
        scratchGm_.SetGlobalBuffer((__gm__ int32_t *)scratchIn);
        idxBaseGm_.SetGlobalBuffer((__gm__ uint8_t *)idxBaseIn);  // host sends A2 lane masks (uint8)

        // ---- UB layout (~89KB; MUST stay under the ~96KB sub-block window) ----
        // L1 layout vs v2: the emb fp32 rows ([8][256]f, 8KB) moved INTO the
        // lg2 buffer (dual-use: the region-1 GEMV and the region-2/epilogue
        // lg2 consumers are mutually exclusive in time), and miscBuf grew
        // 15360 -> 16384 (fixes the v2 latent overflow where ids64_ ended
        // 512B past the misc allocation — a suspected flake source).
        // s16B1 keeps the whole region-1 stripe band.
        const int64_t laneLen = (BLK_V2 > coreNum_ * B_PAD) ? BLK_V2 : coreNum_ * B_PAD;
        pipe_.InitBuffer(w2TBuf_, BLK_V * R_CONST * sizeof(half));        // 8KB
        pipe_.InitBuffer(w2F32Buf_, BLK_V * R_CONST * sizeof(float));     // 16KB
        pipe_.InitBuffer(prodBuf_, BLK_V * R_CONST * sizeof(float));      // 16KB
        pipe_.InitBuffer(s16B1Buf_, B_PAD * sw1_ * sizeof(half));         // 6.6KB
        pipe_.InitBuffer(lg2Buf_, B_PAD * BLK_V2 * sizeof(half));         // 8KB
        const int64_t laneSize = laneLen * 20 + 8192 + 64;
        pipe_.InitBuffer(laneBuf_, laneSize);                             // ~18.2KB
        pipe_.InitBuffer(miscBuf_, 16384);                                // 16KB

        // NOTE: TBuf::Get<T>(len) is a length view; sub-views must use
        // GetWithOffset<T>(elemCount, byteOffset).
        w2T_ = w2TBuf_.Get<half>();
        w2F32_ = w2F32Buf_.Get<float>();
        prod_ = prodBuf_.Get<float>();
        s16B1_ = s16B1Buf_.Get<half>();
        lg2_ = lg2Buf_.Get<half>();
        // L1 layout: embB_ ([8][256] fp32, 8KB) lives in the lg2 buffer —
        // the region-1 GEMV and the region-2/epilogue lg2 consumers are
        // mutually exclusive in time, so the fp32 emb rows and the [8][512]
        // half lg2 tiles share the same 8KB.
        embB_ = lg2Buf_.Get<float>();  // dual-use: [2048]f over the lg2 8KB
        // round-2 Step A views: band = whole region-1 stripe band (s16B1Buf
        // was reserved, unused); epilogue rows = lg2 rows 0/1 (region-2 runs
        // AFTER the epilogue) + dead lane slots idxBaseF_/infF_/redTmpF_
        band16_ = s16B1_;
        lgRowH_ = lg2_;
        s16RowH_ = lg2Buf_.GetWithOffset<half>(512, 7 * BLK_V2 * sizeof(half));  // row 7, clear of the batched B rows
        // lane views (byte offsets, 32B-aligned):
        idxSelF_ = laneBuf_.GetWithOffset<float>(laneLen, 0);                        // [laneLen]
        idxBaseF_ = laneBuf_.GetWithOffset<float>(laneLen, laneLen * 4);             // [laneLen]
        infF_ = laneBuf_.GetWithOffset<float>(laneLen, laneLen * 8);                 // [laneLen]
        redTmpF_ = laneBuf_.GetWithOffset<float>(laneLen, laneLen * 12);             // [laneLen]
        mRow16_ = laneBuf_.GetWithOffset<half>(laneLen, laneLen * 16);               // [laneLen]
        redTmpH_ = mRow16_;  // sequential reuse (ReduceMax tmp then Duplicate dst)
        bRowF32_ = idxBaseF_;   // dead lane slots reused as epilogue rows
        lgRowF32_ = infF_;
        sumRowF32_ = redTmpF_;
        arTmp_ = laneBuf_.GetWithOffset<uint8_t>(8192, laneLen * 20);                // 8KB (ReduceSum AR {16,256} requires 8KB)
        arTmpF_ = laneBuf_.GetWithOffset<float>(2048, laneLen * 20);                 // same storage, RM tmp view
        // misc views (byte offsets, 32B-aligned):
        emb16_ = miscBuf_.GetWithOffset<half>(B_PAD * R_CONST, 0);       // [8][256] 4KB
        lg1Tile_ = miscBuf_.GetWithOffset<half>(B_PAD * BLK_V, 4096);    // [8][16]
        bias16All_ = miscBuf_.GetWithOffset<half>(B_PAD * BLK_V, 4096 + B_PAD * BLK_V * 2);
        s16Tile_ = miscBuf_.GetWithOffset<half>(B_PAD * BLK_V, 4096 + B_PAD * BLK_V * 4);
        bF32_ = miscBuf_.GetWithOffset<float>(B_PAD * BLK_V, 5120);
        lgF32_ = miscBuf_.GetWithOffset<float>(B_PAD * BLK_V, 5120 + B_PAD * BLK_V * 4);
        sumF32_ = miscBuf_.GetWithOffset<float>(B_PAD * BLK_V, 5120 + B_PAD * BLK_V * 8);
        // round-2 fix #3: cross-launch handoffs as FULL 128B MTE3 lines
        // (v1's 32B partial-line stores intermittently left the next
        // launch's MTE2 reads hitting stale L2 lines).
        // candLine [32]f = {v[8], i[8], pad[16]}; tokLine [32]i32 = {tok[8], 0[24]}
        candLineF_ = miscBuf_.GetWithOffset<float>(32, 7072);
        candValF_ = candLineF_;
        candIdxF_ = miscBuf_.GetWithOffset<float>(8, 7104);
        tokLineI_ = miscBuf_.GetWithOffset<int32_t>(32, 7232);
        tokLineF_ = miscBuf_.GetWithOffset<float>(32, 7232);  // same storage, Duplicate view
        tokI32_ = tokLineI_;
        dst2F_ = miscBuf_.GetWithOffset<float>(8, 7392);
        // [1024]f contiguous view spanning the idxBaseF_/infF_ slots (region-2
        // chunk cast dst / region-1 epilogue argmax row; phases are sequential)
        rowF32_ = laneBuf_.GetWithOffset<float>(1024, 2048);
        wsValF_ = miscBuf_.GetWithOffset<float>(coreNum_ * 32, 7424);   // 6KB interleaved {v,i,pad}
        out64_ = miscBuf_.GetWithOffset<int64_t>(72, 7424 + coreNum_ * 128);           // 576B
        scratchAll_ = miscBuf_.GetWithOffset<int32_t>(32 * 9, 7424 + coreNum_ * 128 + 576);  // 1.15KB
        ids64_ = miscBuf_.GetWithOffset<int64_t>(64, 7424 + coreNum_ * 128 + 1792);    // 512B

        Duplicate(infF_, 1e30f, laneLen);
    }


    __aicore__ inline void Process()
    {
        // K+1 launch structure: launch #k (k < K) performs one Markov step and
        // publishes per-core candidates; launch #K reduces the final
        // candidates and writes out_ids. The launch boundary is the grid
        // barrier (validated Triton-reference structure); no in-kernel
        // cross-core SyncAll is needed.
        if (isFinal_ != 0) {
            FinalStage();
            return;
        }
        ResetCandidates();
        ResolveTokens(stepK_);
        GatherEmb();
#if DSPARK_SKIP < 3
        ComputeStripe1(stepK_);
#endif
#if DSPARK_SKIP < 2
        ComputeStripe2(stepK_);
#endif
        WriteCandidate();
    }

private:
    __aicore__ inline void Mte2ToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void VToMte3()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }

    __aicore__ inline void ScalarToMte3()
    {
        // scalar (S-pipe) stores to UB must be complete before MTE3 reads:
        // full-pipeline drain from the S pipe.
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ResetCandidates()
    {
        for (int64_t b = 0; b < B_PAD; ++b) {
            stV0_[b] = -65504.0h;
            stV1_[b] = -65504.0h;
            stI0_[b] = 0.0f;
            stI1_[b] = 0.0f;
        }
    }

    // Lexicographic (val, idx) merge of all per-core candidates (S4),
    // vectorized: RA max -> EQ mask -> select idx / INF -> RA min.
    __aicore__ inline void ReduceCandidatesToTokens()
    {
        // scalar merge with GetValue halving: max-only first pass, idx read
        // deferred to the rare tie (exact v1 lexicographic semantics, S4).
        // (P1's vector chain works but costs a B-independent ~7us -- the
        // 1536-lane Level-2 ops are too heavy per call; archived in R8.)
        const int32_t n = static_cast<int32_t>(coreNum_ * 32);
        DataCopy(wsValF_, wsValGm_, n);
        Mte2ToV();
        PipeBarrier<PIPE_ALL>();
        for (int64_t b = 0; b < B_; ++b) {
            float bestV = -65504.0f;
            float bestI = 0.0f;
            bool tie = false;
            for (int64_t p = 0; p < coreNum_; ++p) {
                const float cv = wsValF_.GetValue(p * 32 + b);
                if (cv > bestV) {
                    bestV = cv;
                    bestI = -1.0f;
                    tie = false;
                } else if (cv == bestV) {
                    tie = true;
                }
            }
            if (bestI < 0.0f || tie) {
                bestI = 1e30f;
                for (int64_t p = 0; p < coreNum_; ++p) {
                    const float cv = wsValF_.GetValue(p * 32 + b);
                    if (cv == bestV) {
                        const float ci = wsValF_.GetValue(p * 32 + 8 + b);
                        if (ci < bestI) { bestI = ci; }
                    }
                }
            }
            tokI32_.SetValue(b, static_cast<int32_t>(bestI));
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ResolveTokens(int64_t k)
    {
        if (k == 0) {
            // every core independently extracts the seed column from the raw
            // int64 ids tensor (no cross-core ordering dependency)
            DataCopy(ids64_, idsGm_, B_ * (K_ + 1));
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            for (int64_t b = 0; b < B_; ++b) {
                tokI32_.SetValue(b, static_cast<int32_t>(ids64_.GetValue(b * (K_ + 1))));
            }
            for (int64_t b = B_; b < B_PAD; ++b) {
                tokI32_.SetValue(b, 0);
            }
            if (pid_ == 0) {
                Duplicate(tokLineF_, 0.0f, 32);
                PipeBarrier<PIPE_V>();
                // re-store the seeds into the zeroed line, then one 128B write
                for (int64_t b = 0; b < B_; ++b) {
                    tokI32_.SetValue(b, static_cast<int32_t>(ids64_.GetValue(b * (K_ + 1))));
                }
                for (int64_t b = B_; b < B_PAD; ++b) {
                    tokI32_.SetValue(b, 0);
                }
                ScalarToMte3();
                DataCopy(scratchGm_, tokLineI_, 32);
                PipeBarrier<PIPE_MTE3>();
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
        ReduceCandidatesToTokens();
        if (pid_ == 0) {
            // stash merged tokens before zeroing the 128B line
            int32_t tk[8];
            for (int64_t b = 0; b < B_; ++b) {
                tk[b] = tokI32_.GetValue(b);
            }
            PipeBarrier<PIPE_ALL>();
            Duplicate(tokLineF_, 0.0f, 32);
            PipeBarrier<PIPE_V>();
            for (int64_t b = 0; b < B_; ++b) {
                tokI32_.SetValue(b, tk[b]);
            }
            ScalarToMte3();
            DataCopy(scratchGm_[k * 32], tokLineI_, 32);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void GatherEmb()
    {
        // fully unrolled gather: explicit scalars, no arrays, no loop vars
        PipeBarrier<PIPE_ALL>();
        const int32_t t0 = tokI32_.GetValue(0);
        const int32_t t1 = (B_ > 1) ? tokI32_.GetValue(1) : 0;
        const int32_t t2 = (B_ > 2) ? tokI32_.GetValue(2) : 0;
        const int32_t t3 = (B_ > 3) ? tokI32_.GetValue(3) : 0;
        const int32_t t4 = (B_ > 4) ? tokI32_.GetValue(4) : 0;
        const int32_t t5 = (B_ > 5) ? tokI32_.GetValue(5) : 0;
        const int32_t t6 = (B_ > 6) ? tokI32_.GetValue(6) : 0;
        // register-cached tokens for the per-(tile,b) GM re-read in
        // BroadcastEmbTile (E5: removes the per-(tile,b) GetValue scalar
        // storm that correlated with wrong tiles)
        curToks_[0] = t0; curToks_[1] = t1; curToks_[2] = t2; curToks_[3] = t3;
        curToks_[4] = t4; curToks_[5] = t5; curToks_[6] = t6; curToks_[7] = 0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(emb16_[0], w1Gm_[static_cast<int64_t>(t0) * R_CONST], R_CONST);
        if (B_ > 1) { DataCopy(emb16_[R_CONST], w1Gm_[static_cast<int64_t>(t1) * R_CONST], R_CONST); }
        if (B_ > 2) { DataCopy(emb16_[2 * R_CONST], w1Gm_[static_cast<int64_t>(t2) * R_CONST], R_CONST); }
        if (B_ > 3) { DataCopy(emb16_[3 * R_CONST], w1Gm_[static_cast<int64_t>(t3) * R_CONST], R_CONST); }
        if (B_ > 4) { DataCopy(emb16_[4 * R_CONST], w1Gm_[static_cast<int64_t>(t4) * R_CONST], R_CONST); }
        if (B_ > 5) { DataCopy(emb16_[5 * R_CONST], w1Gm_[static_cast<int64_t>(t5) * R_CONST], R_CONST); }
        if (B_ > 6) { DataCopy(emb16_[6 * R_CONST], w1Gm_[static_cast<int64_t>(t6) * R_CONST], R_CONST); }
        Mte2ToV();
        PipeBarrier<PIPE_V>();
    }

    // Region 1 [vLo1, vHi1): GEMV into the [B, sw1] band + whole-stripe
    // materialization + ONE first-max argmax per (b, launch).
    //
    // GEMV path: the v2-verified Cast+Mul chain (fp32 product domain, S3).
    // The L1 MulAddDst rework was implemented and measured but REVERTED:
    // on this platform the src1 broadcast tile only stays visible to
    // MulAddDst when re-materialized per (tile, b) via 16x UB->UB DataCopy
    // + a full PIPE_ALL drain (Level-0/Level-2 broadcast APIs, GM re-read,
    // and lighter barriers all lose the tile — dump/bisect verified), and
    // that materialization tax exceeds the P2 microkernel payoff
    // (benchmark 0.616x vs 0.655-0.685x). Findings recorded in trace.md.
    __aicore__ inline void ComputeStripe1(int64_t k)
    {
        if (w1_ <= 0) { return; }
        for (int64_t b = 0; b < B_; ++b) {
            Cast(embB_[b * R_CONST], emb16_[b * R_CONST], RoundMode::CAST_NONE, R_CONST);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t d = 0; d < w1_; d += BLK_V) {
            const int64_t v0 = vLo1_ + d;
            const int64_t curLen = (w1_ - d < BLK_V) ? (w1_ - d) : BLK_V;
            DataCopy(w2T_, w2Gm_[v0 * R_CONST], curLen * R_CONST);
            Mte2ToV();
            Cast(w2F32_, w2T_, RoundMode::CAST_NONE, curLen * R_CONST);
            PipeBarrier<PIPE_V>();
            for (int64_t b = 0; b < B_; ++b) {
                // independent dst rows: issue the whole per-v Mul chain, one
                // barrier before the ReduceSum consumer
                for (int64_t v = 0; v < curLen; ++v) {
                    Mul(prod_[v * R_CONST], w2F32_[v * R_CONST], embB_[b * R_CONST], R_CONST);
                }
                PipeBarrier<PIPE_V>();
                uint32_t rs[2] = {static_cast<uint32_t>(curLen), static_cast<uint32_t>(R_CONST)};
                ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumF32_[b * BLK_V], prod_,
                                                                      arTmp_, rs, true);
                PipeBarrier<PIPE_V>();
                // materialization point 1: fp32 GEMV accumulator -> fp16 (RNE)
                Cast(band16_[b * sw1_ + d], sumF32_[b * BLK_V], RoundMode::CAST_ROUND, curLen);
                // no per-(tile,b) barrier: band is only read at the epilogue;
                // the epilogue's first RAW read is covered by the loop-end drain
            }
        }
        PipeBarrier<PIPE_V>();  // band complete before the epilogue reads it
        // materialization point 2 over the whole stripe band, then ONE
        // first-max argmax per b (S4)
        ComputeStripe1Epilogue(k);
    }

    // whole-band materialization point 2 + one first-max argmax per b (S4)
    __aicore__ inline void ComputeStripe1Epilogue(int64_t k)
    {
        // batch all B logits-row copies into lg2 rows, one MTE2 sync
        for (int64_t b = 0; b < B_; ++b) {
            DataCopy(lg2_[b * BLK_V2], logitsGm_[b * K_ * V_ + k * V_ + vLo1_], w1_);
        }
        Mte2ToV();
        for (int64_t b = 0; b < B_; ++b) {
            Cast(bRowF32_, band16_[b * sw1_], RoundMode::CAST_NONE, w1_);
            PipeBarrier<PIPE_V>();
            Cast(lgRowF32_, lg2_[b * BLK_V2], RoundMode::CAST_NONE, w1_);
            PipeBarrier<PIPE_V>();
            Add(sumRowF32_, bRowF32_, lgRowF32_, w1_);
            PipeBarrier<PIPE_V>();
            Cast(s16RowH_, sumRowF32_, RoundMode::CAST_ROUND, w1_);
            PipeBarrier<PIPE_V>();
            Cast(rowF32_, s16RowH_, RoundMode::CAST_NONE, w1_);
            PipeBarrier<PIPE_V>();
            ReduceMax(dst2F_, rowF32_, redTmpF_, static_cast<int32_t>(w1_), true);
            PipeBarrier<PIPE_V>();  // V drained -> scalar GetValue safe
            const float mbF = dst2F_.GetValue(0);
            const float idxRawF = dst2F_.GetValue(1);
            uint32_t idxBits = 0;
            __builtin_memcpy(&idxBits, &idxRawF, sizeof(uint32_t));
            const int32_t idxSigned = static_cast<int32_t>(idxBits & 0x7FFFFFFFu);
            stV0_[b] = static_cast<half>(mbF);
            stI0_[b] = static_cast<float>(idxSigned) + static_cast<float>(vLo1_);
        }
    }

    // Region 2 [vLo2, vHi2): bias == 0 for v >= Vd, s16 == logits (S5).
    // Round-2 restructure: per-b whole-row copy + fp32 cast + chunked
    // first-max argmax (1024-elem chunks ascend in v; strictly-greater
    // running merge keeps the first max, S4). Replaces the per-tile
    // [8,512] band copies and per-(tile,b) PIPE_ALL scalar epilogues.
    __aicore__ inline void ComputeStripe2(int64_t k)
    {
        if (w2_ <= 0) { return; }
        for (int64_t b = 0; b < B_; ++b) {
            // batch all chunk copies for this b (<= 3 x 1024 halves fits lg2 rows)
            for (int64_t d = 0; d < w2_; d += CHUNK2) {
                const int64_t curLen = (w2_ - d < CHUNK2) ? (w2_ - d) : CHUNK2;
                DataCopy(lg2_[d], logitsGm_[b * K_ * V_ + k * V_ + vLo2_ + d], curLen);
            }
            Mte2ToV();
            for (int64_t d = 0; d < w2_; d += CHUNK2) {
                const int64_t v0 = vLo2_ + d;
                const int64_t curLen = (w2_ - d < CHUNK2) ? (w2_ - d) : CHUNK2;
                Cast(rowF32_, lg2_[d], RoundMode::CAST_NONE, curLen);
                PipeBarrier<PIPE_V>();
                ReduceMax(dst2F_, rowF32_, arTmpF_, static_cast<int32_t>(curLen), true);
                PipeBarrier<PIPE_V>();  // V drained -> scalar GetValue safe (was PIPE_ALL)
                const float mbF = dst2F_.GetValue(0);
                const float idxRawF = dst2F_.GetValue(1);
                uint32_t idxBits = 0;
                __builtin_memcpy(&idxBits, &idxRawF, sizeof(uint32_t));
                const int32_t idxSigned = static_cast<int32_t>(idxBits & 0x7FFFFFFFu);
                const half mb = static_cast<half>(mbF);
                const float tIdx = static_cast<float>(idxSigned) + static_cast<float>(v0);
                if (mbF > static_cast<float>(stV1_[b])) { stV1_[b] = mb; stI1_[b] = tIdx; }
            }
        }
    }

    // Merge the two region candidates (lexicographic) and publish to GM.
    __aicore__ inline void WriteCandidate()
    {
        for (int64_t b = 0; b < B_; ++b) {
            const float v0 = static_cast<float>(stV0_[b]);
            const float v1 = static_cast<float>(stV1_[b]);
            if (v1 > v0) {
                candValF_.SetValue(b, static_cast<float>(v1));
                candIdxF_.SetValue(b, stI1_[b]);
            } else {
                candValF_.SetValue(b, static_cast<float>(v0));
                candIdxF_.SetValue(b, stI0_[b]);
            }
        }
        Duplicate(candLineF_, 0.0f, 32);  // zero the whole 128B line first
        PipeBarrier<PIPE_V>();
        // re-publish the winners (scalar stores land after the Duplicate)
        for (int64_t b = 0; b < B_; ++b) {
            const float v0 = static_cast<float>(stV0_[b]);
            const float v1 = static_cast<float>(stV1_[b]);
            if (v1 > v0) {
                candValF_.SetValue(b, static_cast<float>(v1));
                candIdxF_.SetValue(b, stI1_[b]);
            } else {
                candValF_.SetValue(b, static_cast<float>(v0));
                candIdxF_.SetValue(b, stI0_[b]);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(wsValGm_[pid_ * 32], candLineF_, 32);  // one full 128B line
        // round-2 fix #2: the workspace MTE3 writes must RETIRE before the
        // kernel exits — the launch boundary does not guarantee the MTE3
        // queue is drained, so the next launch's ReduceCandidatesToTokens
        // could otherwise read stale workspace entries (residual ~10% flaky
        // candidates in v1).
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void FinalStage()
    {
        ReduceCandidatesToTokens();
        if (pid_ == 0) {
            int32_t tk[8];
            for (int64_t b = 0; b < B_; ++b) {
                tk[b] = tokI32_.GetValue(b);
            }
            PipeBarrier<PIPE_ALL>();
            Duplicate(tokLineF_, 0.0f, 32);
            PipeBarrier<PIPE_V>();
            for (int64_t b = 0; b < B_; ++b) {
                tokI32_.SetValue(b, tk[b]);
            }
            ScalarToMte3();
            DataCopy(scratchGm_[K_ * 32], tokLineI_, 32);
            PipeBarrier<PIPE_MTE3>();
            DataCopy(scratchAll_, scratchGm_, (K_ + 1) * 32);
            Mte2ToV();
            PipeBarrier<PIPE_ALL>();
            for (int64_t c = 0; c <= K_; ++c) {
                for (int64_t b = 0; b < B_; ++b) {
                    out64_.SetValue(b * (K_ + 1) + c,
                                    static_cast<int64_t>(scratchAll_.GetValue(c * 32 + b)));
                }
            }
            ScalarToMte3();
            DataCopyExtParams outExt;
            outExt.blockCount = 1;
            outExt.blockLen = static_cast<uint32_t>(B_ * (K_ + 1) * sizeof(int64_t));
            outExt.srcStride = 0;
            outExt.dstStride = 0;
            DataCopyPad(outGm_[0], out64_, outExt);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> w2TBuf_, w2F32Buf_, prodBuf_, s16B1Buf_, lg2Buf_;
    TBuf<TPosition::VECCALC> laneBuf_, miscBuf_;

    LocalTensor<half> w2T_, embTile_, lg2_, s16B1_, emb16_, lg1Tile_, bias16All_, s16Tile_, mRow16_, redTmpH_, mOne_, mAll_;
// redTmpH_ aliases mRow16_ (sequential reuse)
    LocalTensor<float> w2F32_, prod_, bF32_, lgF32_, sumF32_, embB_;
    LocalTensor<float> idxBaseF_, infF_, idxSelF_, redTmpF_, tOneF_;
    LocalTensor<uint8_t> cmpMask_, brcTmp_, arTmp_;
    LocalTensor<float> arTmpF_;
    LocalTensor<float> candValF_, candIdxF_, tIdxF_, valMaxF_, wsValF_, wsIdxF_, wsSelF_;
    LocalTensor<float> dst2F_, rowF32_;
    LocalTensor<float> wsAll_, wsSwap_, selVF_, selIF_, selTF_;
    LocalTensor<uint8_t> masksV_, mTie_;
    LocalTensor<float> candLineF_, tokLineF_;
    LocalTensor<int32_t> tokLineI_;
    LocalTensor<half> band16_, lgRowH_, s16RowH_;
    LocalTensor<float> bRowF32_, lgRowF32_, sumRowF32_;
    LocalTensor<int32_t> tokI32_;
    LocalTensor<int32_t> scratchAll_;
    LocalTensor<int64_t> out64_;
    LocalTensor<int64_t> ids64_;

    GlobalTensor<half> logitsGm_, w1Gm_, w2Gm_;
    GlobalTensor<int64_t> idsGm_, outGm_;
    GlobalTensor<int32_t> seed32Gm_;
    GlobalTensor<float> wsValGm_, wsIdxGm_;
    GlobalTensor<uint8_t> idxBaseGm_;
    GlobalTensor<int32_t> scratchGm_;

    int64_t pid_ = 0, B_ = 0, K_ = 0, V_ = 0, Vd_ = 0, sw1_ = 0, sw2_ = 0, coreNum_ = 0;
    int64_t stepK_ = 0;
    int64_t isFinal_ = 0;
    int64_t vLo1_ = 0, vHi1_ = 0, w1_ = 0, vLo2_ = 0, vHi2_ = 0, w2_ = 0;
    half stV0_[8];
    int32_t curToks_[8];  // register-cached tokens (GatherEmb writes, per launch)
    half stV1_[8];
    float stI0_[8];
    float stI1_[8];
};

}  // namespace

extern "C" __global__ __aicore__ void dspark_markov_head(
    GM_ADDR logits, GM_ADDR ids, GM_ADDR seed32, GM_ADDR w1, GM_ADDR w2, GM_ADDR out,
    GM_ADDR ws_val, GM_ADDR ws_idx, GM_ADDR scratch, GM_ADDR idx_base,
    int64_t B, int64_t K, int64_t V, int64_t Vd, int64_t sw1, int64_t sw2, int64_t core_num,
    int64_t step_k, int64_t is_final)
{
    // K+1 launch structure: launch #k (k < K) = one Markov step; launch #K =
    // final reduction + out_ids. The launch boundary is the grid barrier, so
    // no in-kernel SyncAll exists and the cube unit of this __aicore__
    // dual-run kernel can exit immediately (its scalar data paths would
    // otherwise publish garbage candidates into the shared workspace).
#if defined(SPLIT_CORE_CUBE)
    // cube unit: mirrors the barrier sequence so the AIV group's SyncAll
    // completes (1 per step); no real work.
    for (int64_t k = 0; k < K; ++k) {
        AscendC::SyncAll();
    }
    return;
#endif
    DsparkMarkovChain op;
    op.Init(logits, ids, seed32, w1, w2, out, ws_val, ws_idx, scratch, idx_base,
             B, K, V, Vd, sw1, sw2, core_num, step_k, is_final);
    op.Process();
}

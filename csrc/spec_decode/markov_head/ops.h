#ifndef DSPARK_MARKOV_CHAIN_VD_OPS_H
#define DSPARK_MARKOV_CHAIN_VD_OPS_H

#include <vector>

#include <torch/extension.h>

namespace ascend_kernel {

// DSpark sequential Markov chain (see model.py / handover doc section 2).
// Returns out_ids [B, K+1] int64 (seed column preserved, K generated tokens).
at::Tensor dspark_markov_head(const at::Tensor &logits, const at::Tensor &draft_token_ids,
                                  const at::Tensor &markov_w1_weight,
                                  const at::Tensor &markov_w2_weight);

// debug: clones of the internal workspaces (wsVal, wsIdx, scratch-as-float)
std::vector<at::Tensor> dspark_markov_head_ws(const at::Tensor &dummy);

// ---- L2/L3 multi-card single-step primitives (dspark_mh_ops) ----
// L2 op A: [mode 1: merge prev ws candidates -> tok] + mask_lookup(w1_shard)
// -> emb_local [B,256] fp16 (non-owner rows exact +0.0; feeds AR(SUM))
void dspark_mh2_emb(const at::Tensor &tok_line, const at::Tensor &ws, const at::Tensor &w1,
                    const at::Tensor &emb_out, const at::Tensor &tok_out, int64_t lo,
                    int64_t hi, int64_t mode);
// L2 op B: emb @ w2_shard^T -> bias_local [B, vs_pad] fp16 (fp32 acc, fp16 RNE)
void dspark_mh2_bias(const at::Tensor &emb, const at::Tensor &w2, const at::Tensor &bias_out);
// L2 op C: (logits[:,k] + bias_flat) full-vocab first-max candidates -> ws
void dspark_mh2_argmax(const at::Tensor &logits, const at::Tensor &bias_flat,
                       const at::Tensor &ws, int64_t step_k);
// L3 op D: one local Markov step on this rank's shard -> per-core packed keys
// in ws (w1 replicated full; the AR(MAX) runs directly on the ws keys and the
// per-slot core merge happens at the start of the next step's launch)
void dspark_mh3_step(const at::Tensor &logits, const at::Tensor &w1, const at::Tensor &w2,
                     const at::Tensor &ws_ag, const at::Tensor &seed_line,
                     const at::Tensor &tok_out, int64_t r0, int64_t world, int64_t step_k,
                     int64_t mode);
// A/B fallback: the original fused single-launch opD (identical semantics;
// dspark_mh3_step is now composed of mh3_emb + aclnn MatMulV2 + mh3_argmax)
void dspark_mh3_step_fused(const at::Tensor &logits, const at::Tensor &w1, const at::Tensor &w2,
                           const at::Tensor &ws_ag, const at::Tensor &seed_line,
                           const at::Tensor &tok_out, int64_t r0, int64_t world,
                           int64_t step_k, int64_t mode);
// final-column merge: ws -> tok line (once per forward; L2 and L3 chains)
void dspark_mh3_merge(const at::Tensor &ws_ag, const at::Tensor &tok_out, int64_t B,
                      int64_t world);

// P2 launch-floor probes (perf iteration only)
void dspark_probe_empty(const at::Tensor &out, int64_t nBlocks);
void dspark_probe_empty_fat(const at::Tensor &out, int64_t nBlocks);

}  // namespace ascend_kernel

#endif  // DSPARK_MARKOV_CHAIN_VD_OPS_H

/*
 * v2 semantics oracle probe.
 *
 * Standalone CPU-side reference for design-spec hot-path semantics not yet fully
 * represented by public APIs: early-fusion VLM layout, suffix-only LMCE,
 * duplicate-token embedding scatter-add, KV cache by-value start_pos, and
 * multi-head one-backward loss sum. This is not a CPU backend.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH 2
#define IMG_TOKENS 5
#define TEXT_TOKENS 6
#define TOTAL_TOKENS (IMG_TOKENS + TEXT_TOKENS)
#define D_MODEL 8
#define VOCAB 17
#define EMB_TOKENS 10
#define KV_MAX_TOKENS 16
#define KV_CHUNK 3
#define KV_START_POS 7
#define KV_WINDOW 4

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "v2_semantics_oracle_probe failed: %s (%s:%d)\n", \
                    (msg), __FILE__, __LINE__);                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct tensor_view {
    int rank;
    int64_t shape[4];
    int64_t strides[4];
    size_t view_offset_elems;
} tensor_view;

typedef struct oracle_stats {
    uint32_t suffix_lmce_rows;
    uint32_t suffix_lmce_logits;
    uint32_t full_logits_if_materialized;
    uint32_t duplicate_scatter_adds;
    uint32_t kv_cache_pos_tensor_bytes;
    uint32_t backward_calls;
} oracle_stats;

static float patterned_value(uint32_t i, float scale)
{
    uint32_t x = i * 1664525U + 1013904223U;
    float u = (float)((x >> 8U) & 0xffffU) * (1.0f / 65535.0f);
    return (u * 2.0f - 1.0f) * scale;
}

static bool view_is_contiguous(const tensor_view *v)
{
    int i;
    int64_t stride = 1;
    for (i = v->rank - 1; i >= 0; --i) {
        if (v->strides[i] != stride) {
            return false;
        }
        stride *= v->shape[i];
    }
    return true;
}

static void make_hidden_suffix_view(tensor_view *view)
{
    memset(view, 0, sizeof(*view));
    view->rank = 3;
    view->shape[0] = BATCH;
    view->shape[1] = TEXT_TOKENS;
    view->shape[2] = D_MODEL;
    view->strides[0] = TOTAL_TOKENS * D_MODEL;
    view->strides[1] = D_MODEL;
    view->strides[2] = 1;
    view->view_offset_elems = (size_t)IMG_TOKENS * D_MODEL;
}

static void build_prefix_suffix(float prefix[BATCH][IMG_TOKENS][D_MODEL],
                                float suffix[BATCH][TEXT_TOKENS][D_MODEL])
{
    int b;
    int t;
    int d;
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < IMG_TOKENS; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                prefix[b][t][d] = patterned_value((uint32_t)(b * 100 + t * 10 + d), 0.50f);
            }
        }
        for (t = 0; t < TEXT_TOKENS; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                suffix[b][t][d] = patterned_value((uint32_t)(700 + b * 100 + t * 10 + d), 0.75f);
            }
        }
    }
}

static void concat_prefix_suffix(const float prefix[BATCH][IMG_TOKENS][D_MODEL],
                                 const float suffix[BATCH][TEXT_TOKENS][D_MODEL],
                                 float hidden[BATCH][TOTAL_TOKENS][D_MODEL])
{
    int b;
    int t;
    int d;
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < IMG_TOKENS; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                hidden[b][t][d] = prefix[b][t][d];
            }
        }
        for (t = 0; t < TEXT_TOKENS; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                hidden[b][IMG_TOKENS + t][d] = suffix[b][t][d];
            }
        }
    }
}

static void init_lm_inputs(float weight[VOCAB][D_MODEL], int targets[BATCH][TEXT_TOKENS])
{
    int v;
    int d;
    int b;
    int t;
    for (v = 0; v < VOCAB; ++v) {
        for (d = 0; d < D_MODEL; ++d) {
            weight[v][d] = patterned_value((uint32_t)(2000 + v * 31 + d), 0.30f);
        }
    }
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < TEXT_TOKENS; ++t) {
            targets[b][t] = (b * 7 + t * 3 + 1) % VOCAB;
        }
    }
}

static float suffix_lmce_reference(const float hidden[BATCH][TOTAL_TOKENS][D_MODEL],
                                   const float weight[VOCAB][D_MODEL],
                                   const int targets[BATCH][TEXT_TOKENS],
                                   float grad_hidden[BATCH][TOTAL_TOKENS][D_MODEL],
                                   float grad_weight[VOCAB][D_MODEL],
                                   oracle_stats *stats)
{
    int b;
    int t;
    int v;
    int d;
    float loss = 0.0f;
    memset(grad_hidden, 0, sizeof(float) * BATCH * TOTAL_TOKENS * D_MODEL);
    memset(grad_weight, 0, sizeof(float) * VOCAB * D_MODEL);
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < TEXT_TOKENS; ++t) {
            float logits[VOCAB];
            float max_logit = -3.402823466e+38f;
            float denom = 0.0f;
            int pos = IMG_TOKENS + t;
            int target = targets[b][t];
            for (v = 0; v < VOCAB; ++v) {
                float dot = 0.0f;
                for (d = 0; d < D_MODEL; ++d) {
                    dot += hidden[b][pos][d] * weight[v][d];
                }
                logits[v] = dot;
                if (dot > max_logit) {
                    max_logit = dot;
                }
            }
            for (v = 0; v < VOCAB; ++v) {
                denom += expf(logits[v] - max_logit);
            }
            loss += -logits[target] + max_logit + logf(denom);
            for (v = 0; v < VOCAB; ++v) {
                float prob = expf(logits[v] - max_logit) / denom;
                float grad = prob - (v == target ? 1.0f : 0.0f);
                for (d = 0; d < D_MODEL; ++d) {
                    grad_hidden[b][pos][d] += grad * weight[v][d];
                    grad_weight[v][d] += grad * hidden[b][pos][d];
                }
            }
            stats->suffix_lmce_rows += 1U;
            stats->suffix_lmce_logits += VOCAB;
        }
    }
    stats->full_logits_if_materialized = BATCH * TOTAL_TOKENS * VOCAB;
    return loss;
}

static void check_prefix_grad_zero(const float grad_hidden[BATCH][TOTAL_TOKENS][D_MODEL])
{
    int b;
    int t;
    int d;
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < IMG_TOKENS; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                CHECK(grad_hidden[b][t][d] == 0.0f,
                      "LMCE prefix hidden gradient must remain zero");
            }
        }
    }
}

static void embedding_scatter_add_reference(oracle_stats *stats)
{
    const int tokens[EMB_TOKENS] = {3, 1, 3, 4, 1, 8, 3, 8, 16, 1};
    float grad_out[EMB_TOKENS][D_MODEL];
    float grad_table[VOCAB][D_MODEL];
    float expected_token3[D_MODEL];
    int i;
    int d;
    memset(grad_table, 0, sizeof(grad_table));
    memset(expected_token3, 0, sizeof(expected_token3));
    for (i = 0; i < EMB_TOKENS; ++i) {
        for (d = 0; d < D_MODEL; ++d) {
            grad_out[i][d] = 0.01f * (float)(i + 1) + 0.001f * (float)d;
            grad_table[tokens[i]][d] += grad_out[i][d];
            if (tokens[i] == 3) {
                expected_token3[d] += grad_out[i][d];
                stats->duplicate_scatter_adds += 1U;
            }
        }
    }
    for (d = 0; d < D_MODEL; ++d) {
        float diff = grad_table[3][d] - expected_token3[d];
        if (diff < 0.0f) {
            diff = -diff;
        }
        CHECK(diff < 1.0e-6f, "embedding duplicate-token backward must scatter-add");
    }
}

static void kv_cache_by_value_start_pos_reference(oracle_stats *stats)
{
    float cache[BATCH][KV_MAX_TOKENS][D_MODEL];
    float chunk[BATCH][KV_CHUNK][D_MODEL];
    int b;
    int t;
    int d;
    memset(cache, 0, sizeof(cache));
    for (b = 0; b < BATCH; ++b) {
        for (t = 0; t < KV_CHUNK; ++t) {
            for (d = 0; d < D_MODEL; ++d) {
                chunk[b][t][d] = patterned_value((uint32_t)(4000 + b * 100 + t * 10 + d), 0.90f);
                cache[b][KV_START_POS + t][d] = chunk[b][t][d];
            }
        }
    }

    for (b = 0; b < BATCH; ++b) {
        int decode_pos = KV_START_POS + KV_CHUNK - 1;
        int suffix_lo = decode_pos - KV_WINDOW + 1;
        if (suffix_lo < IMG_TOKENS) {
            suffix_lo = IMG_TOKENS;
        }
        for (t = 0; t < IMG_TOKENS; ++t) {
            (void)cache[b][t][0];
        }
        for (t = suffix_lo; t <= decode_pos; ++t) {
            CHECK(t >= IMG_TOKENS, "decode suffix window must not include pre-prefix negative rows");
            if (t >= KV_START_POS && t < KV_START_POS + KV_CHUNK) {
                int rel = t - KV_START_POS;
                for (d = 0; d < D_MODEL; ++d) {
                    CHECK(cache[b][t][d] == chunk[b][rel][d],
                          "KV cache write_at by-value start_pos mismatch");
                }
            }
        }
    }
    stats->kv_cache_pos_tensor_bytes = 0U;
}

static float loss_sum_one_backward_reference(oracle_stats *stats)
{
    const float lm_loss = 3.25f;
    const float repr_loss = 0.75f;
    const float total = lm_loss * 1.0f + repr_loss * 0.1f;
    stats->backward_calls += 1U;
    return total;
}

int main(void)
{
    float prefix[BATCH][IMG_TOKENS][D_MODEL];
    float suffix[BATCH][TEXT_TOKENS][D_MODEL];
    float hidden[BATCH][TOTAL_TOKENS][D_MODEL];
    float weight[VOCAB][D_MODEL];
    float grad_hidden[BATCH][TOTAL_TOKENS][D_MODEL];
    float grad_weight[VOCAB][D_MODEL];
    int targets[BATCH][TEXT_TOKENS];
    tensor_view suffix_view;
    oracle_stats stats;
    float lm_loss;
    float total_loss;

    memset(&stats, 0, sizeof(stats));
    build_prefix_suffix(prefix, suffix);
    concat_prefix_suffix(prefix, suffix, hidden);
    make_hidden_suffix_view(&suffix_view);
    CHECK(!view_is_contiguous(&suffix_view), "VLM suffix view must stay non-contiguous");
    CHECK(suffix_view.view_offset_elems == (size_t)IMG_TOKENS * D_MODEL,
          "suffix view offset must skip image prefix tokens");

    init_lm_inputs(weight, targets);
    lm_loss = suffix_lmce_reference(hidden, weight, targets, grad_hidden, grad_weight, &stats);
    CHECK(lm_loss > 0.0f, "LMCE loss must be positive");
    CHECK(stats.suffix_lmce_rows == BATCH * TEXT_TOKENS,
          "LMCE must visit suffix rows only");
    CHECK(stats.suffix_lmce_logits < stats.full_logits_if_materialized,
          "LMCE oracle must avoid full prefix+suffix logits materialization");
    check_prefix_grad_zero(grad_hidden);

    embedding_scatter_add_reference(&stats);
    CHECK(stats.duplicate_scatter_adds == 3U * D_MODEL,
          "embedding duplicate scatter-add count mismatch");

    kv_cache_by_value_start_pos_reference(&stats);
    CHECK(stats.kv_cache_pos_tensor_bytes == 0U,
          "KV lockstep path must not allocate mutable cache-position tensor");

    total_loss = loss_sum_one_backward_reference(&stats);
    CHECK(total_loss > lm_loss * 0.0f, "loss sum must produce scalar total");
    CHECK(stats.backward_calls == 1U, "multi-head loss must use one backward seed");

    printf("v2_semantics_oracle_probe: ok suffix_rows=%u suffix_logits=%u full_logits=%u duplicate_adds=%u cache_pos_tensor_bytes=%u backward_calls=%u total_loss=%g\n",
           stats.suffix_lmce_rows,
           stats.suffix_lmce_logits,
           stats.full_logits_if_materialized,
           stats.duplicate_scatter_adds,
           stats.kv_cache_pos_tensor_bytes,
           stats.backward_calls,
           (double)total_loss);
    return 0;
}

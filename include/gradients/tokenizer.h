#ifndef GRADIENTS_TOKENIZER_H
#define GRADIENTS_TOKENIZER_H

#include <gradients/status.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_tokenizer gd_tokenizer;

typedef struct gd_tokenizer_config {
    int split_digits;
    int allow_special;
} gd_tokenizer_config;

typedef struct gd_bpe_train_config {
    int vocab_size;
    int min_frequency;
    int split_digits;
    int n_special_tokens;
    const char **special_tokens;
    uint64_t seed;
} gd_bpe_train_config;

gd_status gd_bpe_tokenizer_train(const char **input_paths,
                                 int n_input_paths,
                                 const gd_bpe_train_config *cfg,
                                 gd_tokenizer **out);

gd_status gd_bpe_tokenizer_save(gd_tokenizer *tok, const char *tokenizer_path);

gd_status gd_bpe_tokenizer_load(const char *tokenizer_path,
                                const gd_tokenizer_config *cfg,
                                gd_tokenizer **out);

void gd_tokenizer_destroy(gd_tokenizer *tok);

gd_status gd_tokenizer_encode(gd_tokenizer *tok,
                              const char *text,
                              int32_t **ids_out,
                              int *n_ids_out);

gd_status gd_tokenizer_decode(gd_tokenizer *tok,
                              const int32_t *ids,
                              int n_ids,
                              char **text_out);

gd_status gd_tokenizer_id(gd_tokenizer *tok, const char *special, int32_t *id_out);

int gd_tokenizer_vocab_size(const gd_tokenizer *tok);
uint64_t gd_tokenizer_hash(const gd_tokenizer *tok);

void gd_tokenizer_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TOKENIZER_H */

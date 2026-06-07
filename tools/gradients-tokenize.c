#include <gradients/tokenizer.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gradients-tokenize --tokenizer tokenizer.json --input text.txt "
            "--output tokens.i32 [--no-special]\n");
}

static gd_status read_file(const char *path, char **text_out)
{
    FILE *f;
    long end;
    char *text;
    size_t nread;

    if (path == NULL || text_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *text_out = NULL;
    f = fopen(path, "rb");
    if (f == NULL) {
        return GD_ERR_IO;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    end = ftell(f);
    if (end < 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    text = (char *)malloc((size_t)end + 1U);
    if (text == NULL) {
        (void)fclose(f);
        return GD_ERR_OUT_OF_MEMORY;
    }
    nread = fread(text, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(text);
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fclose(f) != 0) {
        free(text);
        return GD_ERR_IO;
    }
    text[(size_t)end] = '\0';
    *text_out = text;
    return GD_OK;
}

static gd_status write_tokens(const char *path, const int32_t *ids, int n_ids)
{
    FILE *f;
    size_t nwrite;
    if (path == NULL || ids == NULL || n_ids < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        return GD_ERR_IO;
    }
    nwrite = fwrite(ids, sizeof(int32_t), (size_t)n_ids, f);
    if (nwrite != (size_t)n_ids) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fclose(f) != 0) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

int main(int argc, char **argv)
{
    const char *tokenizer_path = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;
    gd_tokenizer_config cfg;
    gd_tokenizer *tok = NULL;
    char *text = NULL;
    int32_t *ids = NULL;
    int n_ids = 0;
    gd_status status;
    int i;

    cfg.split_digits = 1;
    cfg.allow_special = 1;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--no-special") == 0) {
            cfg.allow_special = 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }
    if (tokenizer_path == NULL || input_path == NULL || output_path == NULL) {
        usage(stderr);
        return 2;
    }

    status = gd_bpe_tokenizer_load(tokenizer_path, &cfg, &tok);
    if (status != GD_OK) {
        fprintf(stderr, "load failed: %s\n", gd_status_string(status));
        return 1;
    }
    status = read_file(input_path, &text);
    if (status != GD_OK) {
        fprintf(stderr, "read failed: %s\n", gd_status_string(status));
        gd_tokenizer_destroy(tok);
        return 1;
    }
    status = gd_tokenizer_encode(tok, text, &ids, &n_ids);
    if (status != GD_OK) {
        fprintf(stderr, "encode failed: %s\n", gd_status_string(status));
        free(text);
        gd_tokenizer_destroy(tok);
        return 1;
    }
    status = write_tokens(output_path, ids, n_ids);
    if (status != GD_OK) {
        fprintf(stderr, "write failed: %s\n", gd_status_string(status));
        gd_tokenizer_free(ids);
        free(text);
        gd_tokenizer_destroy(tok);
        return 1;
    }
    printf("{\"tokens\":%d,\"output\":\"%s\"}\n", n_ids, output_path);

    gd_tokenizer_free(ids);
    free(text);
    gd_tokenizer_destroy(tok);
    return 0;
}

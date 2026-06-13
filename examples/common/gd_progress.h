#ifndef GD_PROGRESS_H
#define GD_PROGRESS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
#define GD_PRINTF_FORMAT(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
#define GD_PRINTF_FORMAT(fmt_index, first_arg)
#endif

typedef struct gd_progress {
    FILE *stream;
    bool is_tty;
    bool active;
    unsigned bar_width;
    unsigned row_count;
    unsigned printed_rows;
    char **rows;
} gd_progress;

void gd_progress_init(gd_progress *progress, FILE *stream);
void gd_progress_deinit(gd_progress *progress);
bool gd_progress_is_interactive(const gd_progress *progress);

/* One-shot single-line helpers. */
void gd_progress_update(gd_progress *progress, const char *fmt, ...)
    GD_PRINTF_FORMAT(2, 3);
void gd_progress_bar(gd_progress *progress,
                     const char *label,
                     uint64_t current,
                     uint64_t total,
                     const char *fmt,
                     ...) GD_PRINTF_FORMAT(5, 6);

/* Multi-row builder. Rows are zero-indexed: row 0 is the first terminal row. */
bool gd_progress_set_row_count(gd_progress *progress, unsigned row_count);
void gd_progress_reset_rows(gd_progress *progress);
void gd_progress_rowf(gd_progress *progress, unsigned row, const char *fmt, ...)
    GD_PRINTF_FORMAT(3, 4);
void gd_progress_row_appendf(gd_progress *progress, unsigned row, const char *fmt, ...)
    GD_PRINTF_FORMAT(3, 4);
void gd_progress_row_append_bar(gd_progress *progress,
                                unsigned row,
                                const char *label,
                                uint64_t current,
                                uint64_t total);
void gd_progress_render(gd_progress *progress);

void gd_progress_finish(gd_progress *progress);
void gd_progress_clear(gd_progress *progress);
void gd_progress_message(gd_progress *progress, const char *fmt, ...)
    GD_PRINTF_FORMAT(2, 3);

#undef GD_PRINTF_FORMAT

#endif /* GD_PROGRESS_H */

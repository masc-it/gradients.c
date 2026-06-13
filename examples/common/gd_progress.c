#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gd_progress.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool gd_progress_stream_is_tty(FILE *stream)
{
    const char *term;
    int fd;
    if (stream == NULL) {
        return false;
    }
    fd = fileno(stream);
    if (fd < 0 || isatty(fd) == 0) {
        return false;
    }
    term = getenv("TERM");
    return term == NULL || strcmp(term, "dumb") != 0;
}

static char *gd_progress_vformat(const char *fmt, va_list args)
{
    va_list count_args;
    char *buffer;
    int needed;
    if (fmt == NULL) {
        fmt = "";
    }
    va_copy(count_args, args);
    needed = vsnprintf(NULL, 0, fmt, count_args);
    va_end(count_args);
    if (needed < 0) {
        return NULL;
    }
    buffer = (char *)malloc((size_t)needed + 1U);
    if (buffer == NULL) {
        return NULL;
    }
    (void)vsnprintf(buffer, (size_t)needed + 1U, fmt, args);
    return buffer;
}

static char *gd_progress_format_bar(const gd_progress *progress,
                                    const char *label,
                                    uint64_t current,
                                    uint64_t total)
{
    char bar[65];
    unsigned width;
    unsigned filled = 0U;
    unsigned i;
    double percent = 0.0;
    int needed;
    char *line;
    if (label == NULL) {
        label = "progress";
    }
    width = progress != NULL ? progress->bar_width : 28U;
    if (width == 0U) {
        width = 28U;
    }
    if (width >= sizeof(bar)) {
        width = (unsigned)sizeof(bar) - 1U;
    }
    if (total > 0U) {
        const uint64_t clamped = current < total ? current : total;
        percent = ((double)clamped * 100.0) / (double)total;
        filled = (unsigned)(((double)clamped * (double)width) / (double)total);
        if (filled > width) {
            filled = width;
        }
    }
    for (i = 0U; i < width; ++i) {
        if (i < filled) {
            bar[i] = '=';
        } else if (total > 0U && i == filled && current < total) {
            bar[i] = '>';
        } else {
            bar[i] = '.';
        }
    }
    bar[width] = '\0';

    needed = snprintf(NULL,
                      0,
                      "%s %llu/%llu [%s] %6.2f%%",
                      label,
                      (unsigned long long)current,
                      (unsigned long long)total,
                      bar,
                      percent);
    if (needed < 0) {
        return NULL;
    }
    line = (char *)malloc((size_t)needed + 1U);
    if (line == NULL) {
        return NULL;
    }
    (void)snprintf(line,
                   (size_t)needed + 1U,
                   "%s %llu/%llu [%s] %6.2f%%",
                   label,
                   (unsigned long long)current,
                   (unsigned long long)total,
                   bar,
                   percent);
    return line;
}

static void gd_progress_tty_clear_block(gd_progress *progress)
{
    FILE *stream;
    unsigned i;
    const unsigned rows = progress != NULL ? progress->printed_rows : 0U;
    if (progress == NULL || rows == 0U) {
        return;
    }
    stream = progress->stream != NULL ? progress->stream : stdout;
    fputc('\r', stream);
    if (rows > 1U) {
        (void)fprintf(stream, "\033[%uA", rows - 1U);
    }
    for (i = 0U; i < rows; ++i) {
        fputs("\033[2K", stream);
        if (i + 1U < rows) {
            fputc('\n', stream);
        }
    }
    if (rows > 1U) {
        (void)fprintf(stream, "\033[%uA", rows - 1U);
    }
    fputc('\r', stream);
}

static void gd_progress_render_lines(gd_progress *progress,
                                     const char *const *rows,
                                     unsigned row_count)
{
    FILE *stream;
    unsigned i;
    if (progress == NULL) {
        return;
    }
    stream = progress->stream != NULL ? progress->stream : stdout;
    if (row_count == 0U) {
        gd_progress_clear(progress);
        return;
    }
    if (progress->is_tty) {
        if (progress->active) {
            gd_progress_tty_clear_block(progress);
        }
        for (i = 0U; i < row_count; ++i) {
            fputs("\033[2K", stream);
            fputs(rows != NULL && rows[i] != NULL ? rows[i] : "", stream);
            if (i + 1U < row_count) {
                fputc('\n', stream);
            }
        }
        fflush(stream);
        progress->active = true;
        progress->printed_rows = row_count;
    } else {
        for (i = 0U; i < row_count; ++i) {
            fputs(rows != NULL && rows[i] != NULL ? rows[i] : "", stream);
            fputc('\n', stream);
        }
        fflush(stream);
        progress->active = false;
        progress->printed_rows = 0U;
    }
}

static void gd_progress_render_text(gd_progress *progress, const char *text)
{
    const char *rows[1];
    rows[0] = text != NULL ? text : "";
    gd_progress_render_lines(progress, rows, 1U);
}

static void gd_progress_free_rows(gd_progress *progress)
{
    unsigned i;
    if (progress == NULL || progress->rows == NULL) {
        return;
    }
    for (i = 0U; i < progress->row_count; ++i) {
        free(progress->rows[i]);
    }
    free(progress->rows);
    progress->rows = NULL;
    progress->row_count = 0U;
}

static void gd_progress_row_append_owned(gd_progress *progress, unsigned row, char *text)
{
    char *joined;
    size_t old_len;
    size_t add_len;
    if (progress == NULL || text == NULL) {
        free(text);
        return;
    }
    if (row >= progress->row_count || progress->rows == NULL) {
        free(text);
        return;
    }
    if (progress->rows[row] == NULL) {
        progress->rows[row] = text;
        return;
    }
    old_len = strlen(progress->rows[row]);
    add_len = strlen(text);
    if (add_len > SIZE_MAX - old_len - 1U) {
        free(text);
        return;
    }
    joined = (char *)malloc(old_len + add_len + 1U);
    if (joined == NULL) {
        free(text);
        return;
    }
    memcpy(joined, progress->rows[row], old_len);
    memcpy(joined + old_len, text, add_len + 1U);
    free(progress->rows[row]);
    free(text);
    progress->rows[row] = joined;
}

void gd_progress_init(gd_progress *progress, FILE *stream)
{
    if (progress == NULL) {
        return;
    }
    progress->stream = stream != NULL ? stream : stdout;
    progress->is_tty = gd_progress_stream_is_tty(progress->stream);
    progress->active = false;
    progress->bar_width = 28U;
    progress->row_count = 0U;
    progress->printed_rows = 0U;
    progress->rows = NULL;
}

void gd_progress_deinit(gd_progress *progress)
{
    if (progress == NULL) {
        return;
    }
    gd_progress_free_rows(progress);
    progress->active = false;
    progress->printed_rows = 0U;
}

bool gd_progress_is_interactive(const gd_progress *progress)
{
    return progress != NULL && progress->is_tty;
}

void gd_progress_update(gd_progress *progress, const char *fmt, ...)
{
    char *text;
    va_list args;
    va_start(args, fmt);
    text = gd_progress_vformat(fmt, args);
    va_end(args);
    if (text == NULL) {
        return;
    }
    gd_progress_render_text(progress, text);
    free(text);
}

void gd_progress_bar(gd_progress *progress,
                     const char *label,
                     uint64_t current,
                     uint64_t total,
                     const char *fmt,
                     ...)
{
    char *bar;
    char *suffix;
    char *line;
    int needed;
    va_list args;
    if (progress == NULL) {
        return;
    }
    bar = gd_progress_format_bar(progress, label, current, total);
    if (bar == NULL) {
        return;
    }
    va_start(args, fmt);
    suffix = gd_progress_vformat(fmt, args);
    va_end(args);
    if (suffix == NULL) {
        free(bar);
        return;
    }
    needed = snprintf(NULL, 0, "%s %s", bar, suffix);
    if (needed < 0) {
        free(suffix);
        free(bar);
        return;
    }
    line = (char *)malloc((size_t)needed + 1U);
    if (line == NULL) {
        free(suffix);
        free(bar);
        return;
    }
    (void)snprintf(line, (size_t)needed + 1U, "%s %s", bar, suffix);
    gd_progress_render_text(progress, line);
    free(line);
    free(suffix);
    free(bar);
}

bool gd_progress_set_row_count(gd_progress *progress, unsigned row_count)
{
    char **rows;
    if (progress == NULL) {
        return false;
    }
    if (row_count == progress->row_count) {
        return true;
    }
    if (row_count == 0U) {
        gd_progress_free_rows(progress);
        return true;
    }
    rows = (char **)calloc((size_t)row_count, sizeof(rows[0]));
    if (rows == NULL) {
        return false;
    }
    gd_progress_free_rows(progress);
    progress->rows = rows;
    progress->row_count = row_count;
    return true;
}

void gd_progress_reset_rows(gd_progress *progress)
{
    unsigned i;
    if (progress == NULL || progress->rows == NULL) {
        return;
    }
    for (i = 0U; i < progress->row_count; ++i) {
        free(progress->rows[i]);
        progress->rows[i] = NULL;
    }
}

void gd_progress_rowf(gd_progress *progress, unsigned row, const char *fmt, ...)
{
    char *text;
    va_list args;
    if (progress == NULL || row >= progress->row_count || progress->rows == NULL) {
        return;
    }
    va_start(args, fmt);
    text = gd_progress_vformat(fmt, args);
    va_end(args);
    if (text == NULL) {
        return;
    }
    free(progress->rows[row]);
    progress->rows[row] = text;
}

void gd_progress_row_appendf(gd_progress *progress, unsigned row, const char *fmt, ...)
{
    char *text;
    va_list args;
    va_start(args, fmt);
    text = gd_progress_vformat(fmt, args);
    va_end(args);
    gd_progress_row_append_owned(progress, row, text);
}

void gd_progress_row_append_bar(gd_progress *progress,
                                unsigned row,
                                const char *label,
                                uint64_t current,
                                uint64_t total)
{
    gd_progress_row_append_owned(progress,
                                 row,
                                 gd_progress_format_bar(progress, label, current, total));
}

void gd_progress_render(gd_progress *progress)
{
    if (progress == NULL) {
        return;
    }
    gd_progress_render_lines(progress, (const char *const *)progress->rows, progress->row_count);
}

void gd_progress_finish(gd_progress *progress)
{
    FILE *stream;
    if (progress == NULL || !progress->active || !progress->is_tty) {
        return;
    }
    stream = progress->stream != NULL ? progress->stream : stdout;
    fputc('\n', stream);
    fflush(stream);
    progress->active = false;
    progress->printed_rows = 0U;
}

void gd_progress_clear(gd_progress *progress)
{
    FILE *stream;
    if (progress == NULL || !progress->active || !progress->is_tty) {
        return;
    }
    stream = progress->stream != NULL ? progress->stream : stdout;
    gd_progress_tty_clear_block(progress);
    fflush(stream);
    progress->active = false;
    progress->printed_rows = 0U;
}

void gd_progress_message(gd_progress *progress, const char *fmt, ...)
{
    FILE *stream;
    char *text;
    va_list args;
    if (progress == NULL) {
        return;
    }
    gd_progress_clear(progress);
    stream = progress->stream != NULL ? progress->stream : stdout;
    va_start(args, fmt);
    text = gd_progress_vformat(fmt, args);
    va_end(args);
    if (text == NULL) {
        return;
    }
    fputs(text, stream);
    fputc('\n', stream);
    fflush(stream);
    progress->active = false;
    progress->printed_rows = 0U;
    free(text);
}

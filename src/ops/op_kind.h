#ifndef GD_OP_KIND_H
#define GD_OP_KIND_H

/* Generated in the full operator-capsule build. Keep IDs stable across builds. */
typedef enum gd_op_kind {
    GD_OP_INVALID = 0,
    GD_OP_MATMUL = 1,
    GD_OP_LINEAR = 2,
    GD_OP_COUNT = 3,
} gd_op_kind;

#endif /* GD_OP_KIND_H */

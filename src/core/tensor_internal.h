#ifndef GRADIENTS_TENSOR_INTERNAL_H
#define GRADIENTS_TENSOR_INTERNAL_H

#include <stdbool.h>

#include "gradients/graph.h"
#include "gradients/tensor.h"
#include "refcount.h"

struct gd_tensor {
    gd_refcount refcount;
    gd_tensor_desc desc;
    gd_storage *storage;
    gd_graph *graph;
    int value_id;
    bool requires_grad;
    gd_tensor *grad;
    char *name;
};

gd_status _gd_tensor_create_virtual(gd_graph *graph,
                                    int value_id,
                                    const gd_tensor_desc *desc,
                                    gd_tensor **out);

bool _gd_tensor_is_virtual(const gd_tensor *tensor);
int _gd_tensor_value_id(const gd_tensor *tensor);
gd_graph *_gd_tensor_graph(const gd_tensor *tensor);
const gd_tensor_desc *_gd_tensor_desc_ptr(const gd_tensor *tensor);
bool _gd_tensor_is_contiguous(const gd_tensor *tensor);
gd_status _gd_tensor_materialize_from_graph(gd_context *ctx, gd_tensor *tensor);
gd_status _gd_tensor_ensure_grad(gd_context *ctx, gd_tensor *tensor, gd_tensor **grad_out);
gd_status _gd_tensor_zero(gd_tensor *tensor);

#endif /* GRADIENTS_TENSOR_INTERNAL_H */

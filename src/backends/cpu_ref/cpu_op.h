#ifndef GRADIENTS_CPU_OP_H
#define GRADIENTS_CPU_OP_H

#include "cpu_backend.h"

#include "../../core/internal.h"
#include "../../ops/op_impl.h"
#include "../backend.h"

typedef struct _gd_executable _gd_cpu_exec;

typedef gd_status (*_gd_cpu_support_fn)(const _gd_node *node);
typedef gd_status (*_gd_cpu_run_fn)(_gd_cpu_exec *exec, const _gd_node *node);

typedef struct _gd_cpu_op {
    _gd_op_kind kind;
    const char *name;
    _gd_cpu_support_fn support;
    _gd_cpu_run_fn run;
} _gd_cpu_op;

const _gd_cpu_op *_gd_cpu_op_for(_gd_op_kind kind);
gd_status _gd_cpu_support_default(const _gd_node *node);

gd_status _gd_cpu_exec_value(_gd_cpu_exec *exec,
                             int value_id,
                             void **data_out,
                             const gd_tensor_desc **desc_out);
gd_status _gd_cpu_exec_input(_gd_cpu_exec *exec,
                             const _gd_node *node,
                             int input_index,
                             void **data_out,
                             const gd_tensor_desc **desc_out);
gd_status _gd_cpu_exec_output(_gd_cpu_exec *exec,
                              const _gd_node *node,
                              int output_index,
                              void **data_out,
                              const gd_tensor_desc **desc_out);
uint64_t _gd_cpu_exec_run_id(const _gd_cpu_exec *exec);

gd_status _gd_cpu_require_f32(const gd_tensor_desc *desc);

#endif /* GRADIENTS_CPU_OP_H */

#include "gradients/gradients.h"

#define GD_REF(fn, ret, args) \
    static ret (*volatile ref_##fn) args __attribute__((unused)) = fn

/* status.h */
GD_REF(gd_status_name, const char *, (gd_status));
GD_REF(gd_status_message, const char *, (gd_status));
GD_REF(gd_last_error, const char *, (void));

/* device.h */
GD_REF(gd_device_equal, bool, (gd_device, gd_device));
GD_REF(gd_device_type_name, const char *, (gd_device_type));

/* dtype.h */
GD_REF(gd_compute_policy_default, gd_compute_policy, (void));
GD_REF(gd_dtype_sizeof, size_t, (gd_dtype));
GD_REF(gd_dtype_name, const char *, (gd_dtype));

/* context.h */
GD_REF(gd_context_create, gd_status, (gd_context **));
GD_REF(gd_context_destroy, void, (gd_context *));
GD_REF(gd_context_set_default_device, gd_status, (gd_context *, gd_device));
GD_REF(gd_context_default_device, gd_device, (const gd_context *));
GD_REF(gd_context_set_fallback_policy, gd_status, (gd_context *, gd_fallback_policy));
GD_REF(gd_context_fallback_policy, gd_fallback_policy, (const gd_context *));
GD_REF(gd_context_set_compute_policy, gd_status, (gd_context *, gd_compute_policy));
GD_REF(gd_context_compute_policy, gd_compute_policy, (const gd_context *));
GD_REF(gd_synchronize, gd_status, (gd_context *, gd_device));

/* quant.h */
GD_REF(gd_quant_register_format, gd_status, (gd_context *, const gd_quant_format *));
GD_REF(gd_quant_find_format, const gd_quant_format *, (gd_context *, const char *));
GD_REF(gd_quant_desc_create, gd_status, (gd_context *, const gd_quant_format *, int, int, gd_dtype, gd_tensor *, gd_tensor *, const void *, size_t, gd_quant_desc **));
GD_REF(gd_quant_desc_retain, gd_status, (gd_quant_desc *));
GD_REF(gd_quant_desc_release, void, (gd_quant_desc *));
GD_REF(gd_quantize, gd_status, (gd_context *, gd_tensor *, const gd_quant_desc *, gd_tensor **));
GD_REF(gd_dequantize, gd_status, (gd_context *, gd_tensor *, gd_dtype, gd_tensor **));

/* tensor.h */
GD_REF(gd_tensor_desc_contiguous, gd_status, (gd_dtype, gd_device, int, const int64_t *, gd_tensor_desc *));
GD_REF(gd_tensor_desc_nbytes, gd_status, (const gd_tensor_desc *, size_t *, size_t *));
GD_REF(gd_storage_create, gd_status, (gd_context *, const gd_storage_desc *, gd_storage **));
GD_REF(gd_storage_retain, gd_status, (gd_storage *));
GD_REF(gd_storage_release, void, (gd_storage *));
GD_REF(gd_storage_data_cpu, gd_status, (gd_storage *, void **));
GD_REF(gd_storage_copy_from_cpu, gd_status, (gd_context *, gd_storage *, size_t, const void *, size_t));
GD_REF(gd_storage_copy_to_cpu, gd_status, (gd_context *, gd_storage *, size_t, void *, size_t));
GD_REF(gd_storage_nbytes, size_t, (const gd_storage *));
GD_REF(gd_storage_device, gd_device, (const gd_storage *));
GD_REF(gd_tensor_empty, gd_status, (gd_context *, const gd_tensor_desc *, gd_tensor **));
GD_REF(gd_tensor_from_storage, gd_status, (gd_context *, gd_storage *, const gd_tensor_desc *, gd_tensor **));
GD_REF(gd_tensor_retain, gd_status, (gd_tensor *));
GD_REF(gd_tensor_release, void, (gd_tensor *));
GD_REF(gd_tensor_copy_from_cpu, gd_status, (gd_context *, gd_tensor *, const void *, size_t));
GD_REF(gd_tensor_copy_to_cpu, gd_status, (gd_context *, gd_tensor *, void *, size_t));
GD_REF(gd_tensor_ndim, int, (const gd_tensor *));
GD_REF(gd_tensor_size, int64_t, (const gd_tensor *, int));
GD_REF(gd_tensor_stride, int64_t, (const gd_tensor *, int));
GD_REF(gd_tensor_dtype, gd_dtype, (const gd_tensor *));
GD_REF(gd_tensor_device, gd_device, (const gd_tensor *));
GD_REF(gd_tensor_layout, gd_layout, (const gd_tensor *));
GD_REF(gd_tensor_storage, gd_storage *, (const gd_tensor *));
GD_REF(gd_tensor_quant, const gd_quant_desc *, (const gd_tensor *));
GD_REF(gd_tensor_view, gd_status, (gd_tensor *, const gd_tensor_desc *, gd_tensor **));
GD_REF(gd_tensor_reshape, gd_status, (gd_tensor *, int, const int64_t *, gd_tensor **));
GD_REF(gd_tensor_transpose, gd_status, (gd_tensor *, int, int, gd_tensor **));
GD_REF(gd_tensor_slice, gd_status, (gd_tensor *, int, int64_t, int64_t, gd_tensor **));
GD_REF(gd_tensor_contiguous, gd_status, (gd_context *, gd_tensor *, gd_tensor **));
GD_REF(gd_tensor_set_requires_grad, gd_status, (gd_tensor *, bool));
GD_REF(gd_tensor_requires_grad, bool, (const gd_tensor *));
GD_REF(gd_tensor_grad, gd_status, (gd_tensor *, gd_tensor **));

/* graph.h */
GD_REF(gd_graph_create, gd_status, (gd_context *, gd_graph **));
GD_REF(gd_graph_destroy, gd_status, (gd_graph *));
GD_REF(gd_graph_begin, gd_status, (gd_context *, gd_graph *));
GD_REF(gd_graph_end, gd_status, (gd_context *));
GD_REF(gd_graph_validate, gd_status, (gd_graph *));
GD_REF(gd_graph_compile, gd_status, (gd_graph *, gd_device));
GD_REF(gd_graph_run, gd_status, (gd_graph *));
GD_REF(gd_graph_reset, gd_status, (gd_graph *));
GD_REF(gd_graph_dump, gd_status, (gd_graph *, gd_dump_format, const char *));
GD_REF(gd_graph_run_immediate, gd_status, (gd_context *, gd_device, gd_immediate_build_fn, void *));
GD_REF(gd_graph_run_until, gd_status, (gd_graph *, int));
GD_REF(gd_graph_compare, gd_status, (gd_graph *, gd_device, gd_device, const gd_compare_options *));
GD_REF(gd_scope_push, gd_status, (gd_context *, const char *));
GD_REF(gd_scope_pop, gd_status, (gd_context *));
GD_REF(gd_tensor_set_name, gd_status, (gd_tensor *, const char *));
GD_REF(gd_tensor_materialize, gd_status, (gd_context *, gd_tensor *));
GD_REF(gd_tensor_to_cpu, gd_status, (gd_context *, gd_tensor *, void *, size_t));
GD_REF(gd_debug_print_tensor, gd_status, (gd_context *, gd_tensor *, int));
GD_REF(gd_assert_finite, gd_status, (gd_context *, gd_tensor *));
GD_REF(gd_assert_close, gd_status, (gd_context *, gd_tensor *, gd_tensor *, float, float));

/* ops.h */
GD_REF(gd_add, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_mul, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_scale, gd_status, (gd_context *, gd_tensor *, float, gd_tensor **));
GD_REF(gd_matmul, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_matmul_ex, gd_status, (gd_context *, const gd_matmul_desc *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_linear, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_linear_ex, gd_status, (gd_context *, const gd_linear_desc *, gd_tensor *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_relu, gd_status, (gd_context *, gd_tensor *, gd_tensor **));
GD_REF(gd_silu, gd_status, (gd_context *, gd_tensor *, gd_tensor **));
GD_REF(gd_powlu, gd_status, (gd_context *, gd_tensor *, gd_tensor *, float, gd_tensor **));
GD_REF(gd_sum, gd_status, (gd_context *, gd_tensor *, int, bool, gd_tensor **));
GD_REF(gd_mean, gd_status, (gd_context *, gd_tensor *, int, bool, gd_tensor **));
GD_REF(gd_rms_norm, gd_status, (gd_context *, gd_tensor *, gd_tensor *, float, gd_tensor **));
GD_REF(gd_softmax, gd_status, (gd_context *, gd_tensor *, int, gd_tensor **));
GD_REF(gd_cross_entropy, gd_status, (gd_context *, gd_tensor *, gd_tensor *, int, gd_tensor **));
GD_REF(gd_lm_cross_entropy, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_cast, gd_status, (gd_context *, gd_tensor *, gd_dtype, gd_tensor **));
GD_REF(gd_gelu, gd_status, (gd_context *, gd_tensor *, bool, gd_tensor **));
GD_REF(gd_transpose, gd_status, (gd_context *, gd_tensor *, const int *, int, gd_tensor **));
GD_REF(gd_embedding, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_rope, gd_status, (gd_context *, gd_tensor *, gd_tensor *, const gd_rope_config *, gd_tensor **));
GD_REF(gd_sdpa, gd_status, (gd_context *, gd_tensor *, gd_tensor *, gd_tensor *, gd_tensor *, const gd_sdpa_config *, gd_tensor **));
GD_REF(gd_backward, gd_status, (gd_context *, gd_tensor *));
GD_REF(gd_zero_grad, gd_status, (gd_context *, gd_tensor **, int));
GD_REF(gd_clip_grad_norm, gd_status, (gd_context *, gd_tensor **, int, float, gd_tensor **));

/* module.h */
GD_REF(gd_module_create, gd_status, (gd_context *, const char *, gd_module **));
GD_REF(gd_module_destroy, void, (gd_module *));
GD_REF(gd_module_param, gd_status, (gd_module *, const char *, gd_tensor *));
GD_REF(gd_module_child, gd_status, (gd_module *, const char *, gd_module *));
GD_REF(gd_module_parameters, gd_status, (gd_module *, gd_tensor ***, int *));
GD_REF(gd_module_zero_grad, gd_status, (gd_context *, gd_module *));
GD_REF(gd_module_save, gd_status, (gd_module *, const char *));
GD_REF(gd_module_load, gd_status, (gd_module *, const char *, bool));

/* optim.h */
GD_REF(gd_adamw_create, gd_status, (gd_context *, gd_tensor **, int, const gd_adamw_config *, gd_optimizer **));
GD_REF(gd_adamw_create_groups, gd_status, (gd_context *, const gd_param_group *, int, const gd_adamw_config *, gd_optimizer **));
GD_REF(gd_optimizer_destroy, void, (gd_optimizer *));
GD_REF(gd_optimizer_step, gd_status, (gd_context *, gd_optimizer *));
GD_REF(gd_optimizer_step_lr, gd_status, (gd_context *, gd_optimizer *, gd_tensor *));
GD_REF(gd_optimizer_step_amp, gd_status, (gd_context *, gd_optimizer *, gd_amp_scaler *));
GD_REF(gd_optimizer_zero_grad, gd_status, (gd_context *, gd_optimizer *));
GD_REF(gd_optimizer_save, gd_status, (gd_optimizer *, const char *));
GD_REF(gd_optimizer_load, gd_status, (gd_optimizer *, const char *));
GD_REF(gd_lr_scheduler_value, gd_status, (const gd_lr_scheduler_config *, int, float *));
GD_REF(gd_lr_scheduler_write, gd_status, (gd_context *, const gd_lr_scheduler_config *, int, gd_tensor *, float *));
GD_REF(gd_amp_scaler_create, gd_status, (gd_context *, const gd_amp_scaler_config *, gd_amp_scaler **));
GD_REF(gd_amp_scaler_destroy, void, (gd_amp_scaler *));
GD_REF(gd_amp_scaler_scale, float, (const gd_amp_scaler *));
GD_REF(gd_amp_scaler_scale_loss, gd_status, (gd_context *, gd_amp_scaler *, gd_tensor *, gd_tensor **));
GD_REF(gd_amp_scaler_update, gd_status, (gd_context *, gd_amp_scaler *, bool *));
GD_REF(gd_amp_scaler_found_inf, gd_status, (gd_context *, gd_amp_scaler *, bool *));

/* nn.h */
GD_REF(gd_gpt_create, gd_status, (gd_context *, const gd_gpt_config *, uint64_t, gd_gpt **));
GD_REF(gd_gpt_destroy, void, (gd_gpt *));
GD_REF(gd_gpt_parameters, gd_status, (gd_gpt *, gd_tensor ***, int *));
GD_REF(gd_gpt_parameter_groups, gd_status, (gd_gpt *, float, gd_param_group **, int *));
GD_REF(gd_gpt_parameter_groups_free, void, (gd_param_group *, int));
GD_REF(gd_gpt_forward, gd_status, (gd_context *, gd_gpt *, gd_tensor *, gd_tensor *, gd_tensor **));
GD_REF(gd_gpt_forward_loss, gd_status, (gd_context *, gd_gpt *, gd_tensor *, gd_tensor *, gd_tensor *, gd_tensor **));

/* tokenizer.h */
GD_REF(gd_bpe_tokenizer_train, gd_status, (const char **, int, const gd_bpe_train_config *, gd_tokenizer **));
GD_REF(gd_bpe_tokenizer_save, gd_status, (gd_tokenizer *, const char *));
GD_REF(gd_bpe_tokenizer_load, gd_status, (const char *, const gd_tokenizer_config *, gd_tokenizer **));
GD_REF(gd_tokenizer_destroy, void, (gd_tokenizer *));
GD_REF(gd_tokenizer_encode, gd_status, (gd_tokenizer *, const char *, int32_t **, int *));
GD_REF(gd_tokenizer_decode, gd_status, (gd_tokenizer *, const int32_t *, int, char **));
GD_REF(gd_tokenizer_id, gd_status, (gd_tokenizer *, const char *, int32_t *));
GD_REF(gd_tokenizer_vocab_size, int, (const gd_tokenizer *));
GD_REF(gd_tokenizer_hash, uint64_t, (const gd_tokenizer *));
GD_REF(gd_tokenizer_free, void, (void *));

/* dataset.h */
GD_REF(gd_dataset_build, gd_status, (const gd_dataset_build_config *, gd_dataset_build_result *));
GD_REF(gd_dataset_build_result_clear, void, (gd_dataset_build_result *));
GD_REF(gd_gdtok_read_header, gd_status, (const char *, gd_gdtok_header *));

/* dataloader.h */
GD_REF(gd_token_dataset_open, gd_status, (const char **, int, gd_token_dataset **));
GD_REF(gd_token_dataset_close, void, (gd_token_dataset *));
GD_REF(gd_token_dataset_num_samples, uint64_t, (const gd_token_dataset *));
GD_REF(gd_token_dataset_block_len, uint32_t, (const gd_token_dataset *));
GD_REF(gd_token_dataset_vocab_size, uint32_t, (const gd_token_dataset *));
GD_REF(gd_token_dataset_tokenizer_hash, uint64_t, (const gd_token_dataset *));
GD_REF(gd_dataloader_create, gd_status, (gd_context *, gd_token_dataset *, const gd_dataloader_config *, gd_dataloader **));
GD_REF(gd_dataloader_destroy, void, (gd_dataloader *));
GD_REF(gd_dataloader_prefetch, gd_status, (gd_dataloader *));
GD_REF(gd_dataloader_next, gd_status, (gd_dataloader *, gd_batch_slot **));
GD_REF(gd_dataloader_release_slot, gd_status, (gd_dataloader *, gd_batch_slot *));
GD_REF(gd_dataloader_state_save, gd_status, (gd_dataloader *, const char *));
GD_REF(gd_dataloader_state_load, gd_status, (gd_dataloader *, const char *));
GD_REF(gd_dataloader_metrics_get, void, (const gd_dataloader *, gd_dataloader_metrics *));

int main(void)
{
    return 0;
}

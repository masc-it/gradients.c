#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_module failed: %s (%s:%d)\n", (msg),       \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static gd_memory_config module_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 16384U;
    cfg.state_bytes = 4096U;
    cfg.scratch_slot_bytes = 4096U;
    cfg.data_slot_bytes = 4096U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool path_is(const gd_param_set *set, uint32_t index, const char *path)
{
    return set != NULL && index < set->count &&
           strcmp(set->items[index].path, path) == 0;
}

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void test_linear_child_module(gd_context *ctx)
{
    gd_module root;
    gd_linear_layer fc1;
    gd_linear_layer fc2;
    gd_linear_layer_config fc1_cfg;
    gd_linear_layer_config fc2_cfg;
    gd_param_set params;
    gd_param_group groups[2];
    gd_tensor x;
    gd_tensor y;
    int64_t x_shape[2] = {4, 2};
    float fc1_weight[2U * 4U];
    float fc1_bias[4U];
    uint32_t i;

    fc1_cfg = gd_linear_layer_config_make(2, 4, GD_DTYPE_F16, 11U);
    fc2_cfg = gd_linear_layer_config_make(4, 1, GD_DTYPE_F16, 22U);

    CHECK_OK(gd_module_init(ctx, &root, "xor"));
    CHECK_OK(gd_linear_layer_init_child(ctx, &root, "fc1", &fc1, &fc1_cfg));
    CHECK_OK(gd_linear_layer_init_child(ctx, &root, "fc2", &fc2, &fc2_cfg));

    groups[0] = gd_param_group_build("hidden", "xor.fc1.*", 1.0f, 0.0f, true);
    groups[1] = gd_param_group_build("head", "xor.fc2.*", 1.0f, 0.0f, true);

    CHECK_OK(gd_module_collect_params(ctx, &root, groups, GD_ARRAY_LEN(groups), &params));
    CHECK(params.count == 4U, "xor model exposes four params");
    CHECK(path_is(&params, 0U, "xor.fc1.weight"), "fc1 weight path");
    CHECK(path_is(&params, 1U, "xor.fc1.bias"), "fc1 bias path");
    CHECK(path_is(&params, 2U, "xor.fc2.weight"), "fc2 weight path");
    CHECK(path_is(&params, 3U, "xor.fc2.bias"), "fc2 bias path");
    CHECK(params.items[0].group_index == 0 && params.items[2].group_index == 1,
          "param groups matched by path");
    gd_param_set_free(&params);

    CHECK_OK(gd_module_init_params_uniform(ctx, &root, "xor.fc1.weight", -1.0f, 1.0f, 42U));
    CHECK_OK(gd_tensor_read_f32(ctx, &fc1.weight, fc1_weight, GD_ARRAY_LEN(fc1_weight)));
    for (i = 0U; i < GD_ARRAY_LEN(fc1_weight); ++i) {
        CHECK(fc1_weight[i] >= -1.0f && fc1_weight[i] <= 1.0f,
              "module uniform init stays in range");
    }
    CHECK_OK(gd_tensor_one_(ctx, &fc1.bias));
    CHECK_OK(gd_module_init_params_zero(ctx, &root, "xor.fc1.bias"));
    CHECK_OK(gd_tensor_read_f32(ctx, &fc1.bias, fc1_bias, GD_ARRAY_LEN(fc1_bias)));
    for (i = 0U; i < GD_ARRAY_LEN(fc1_bias); ++i) {
        CHECK(abs_f32(fc1_bias[i]) <= 1.0e-6f, "module zero init clears matching bias");
    }

    CHECK_OK(gd_module_freeze(&root, "xor.fc1.*"));
    CHECK_OK(gd_module_parameters(ctx, &root, &params));
    CHECK(!params.items[0].trainable && !params.items[1].trainable,
          "freeze marks hidden layer params non-trainable");
    CHECK(params.items[2].trainable && params.items[3].trainable,
          "freeze leaves head params trainable");
    gd_param_set_free(&params);

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_DATA, GD_DTYPE_F16, 2U, x_shape, 256U, &x));
    CHECK_OK(gd_linear_layer_forward(ctx, &fc1, &x, &y));
    CHECK(y.rank == 2U && y.shape[0] == 4 && y.shape[1] == 4,
          "linear layer forward output shape");
    CHECK_OK(gd_end(ctx));

    gd_param_set_free(&params);
    gd_linear_layer_deinit(&fc2);
    gd_linear_layer_deinit(&fc1);
    gd_module_deinit(&root);
}

static void test_module_list_paths(gd_context *ctx)
{
    gd_module root;
    gd_module_list layers;
    gd_linear_layer layer0;
    gd_linear_layer layer1;
    gd_linear_layer_config config;
    gd_param_set params;

    config = gd_linear_layer_config_make(2, 2, GD_DTYPE_F16, 33U);

    CHECK_OK(gd_module_init(ctx, &root, "toy"));
    CHECK_OK(gd_module_list_init_child(ctx, &root, "layers", &layers, 2U));
    CHECK_OK(gd_linear_layer_init(ctx, &layer0, "linear", &config));
    CHECK_OK(gd_linear_layer_init(ctx, &layer1, "linear", &config));
    CHECK_OK(gd_module_list_set(&layers, 0U, &layer0.mod));
    CHECK_OK(gd_module_list_set(&layers, 1U, &layer1.mod));

    CHECK_STATUS(gd_module_list_set(&layers, 2U, &layer1.mod), GD_ERR_INVALID_ARGUMENT);

    CHECK_OK(gd_module_parameters(ctx, &root, &params));
    CHECK(params.count == 4U, "module list exposes params recursively");
    CHECK(path_is(&params, 0U, "toy.layers.0.weight"), "module list index 0 weight path");
    CHECK(path_is(&params, 2U, "toy.layers.1.weight"), "module list index 1 weight path");
    gd_param_set_free(&params);

    gd_linear_layer_deinit(&layer1);
    gd_linear_layer_deinit(&layer0);
    gd_module_list_deinit(&layers);
    gd_module_deinit(&root);
}

static void test_module_dict_paths(gd_context *ctx)
{
    gd_module root;
    gd_module_dict heads;
    gd_linear_layer lm;
    gd_linear_layer repr;
    gd_linear_layer_config lm_config;
    gd_linear_layer_config repr_config;
    gd_param_set params;

    lm_config = gd_linear_layer_config_make(4, 8, GD_DTYPE_F16, 44U);
    repr_config = gd_linear_layer_config_make(4, 2, GD_DTYPE_F16, 55U);

    CHECK_OK(gd_module_init(ctx, &root, "vlm"));
    CHECK_OK(gd_module_dict_init_child(ctx, &root, "heads", &heads));
    CHECK_OK(gd_linear_layer_init(ctx, &lm, "linear", &lm_config));
    CHECK_OK(gd_linear_layer_init(ctx, &repr, "linear", &repr_config));
    CHECK_OK(gd_module_dict_set(&heads, "lm", &lm.mod));
    CHECK_OK(gd_module_dict_set(&heads, "repr", &repr.mod));

    CHECK_OK(gd_module_parameters(ctx, &root, &params));
    CHECK(params.count == 4U, "module dict exposes params recursively");
    CHECK(path_is(&params, 0U, "vlm.heads.lm.weight"), "module dict lm path");
    CHECK(path_is(&params, 2U, "vlm.heads.repr.weight"), "module dict repr path");
    gd_param_set_free(&params);

    gd_linear_layer_deinit(&repr);
    gd_linear_layer_deinit(&lm);
    gd_module_dict_deinit(&heads);
    gd_module_deinit(&root);
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = module_config();
    gd_status st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_module: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);

    test_linear_child_module(ctx);
    test_module_list_paths(ctx);
    test_module_dict_paths(ctx);

    gd_context_destroy(ctx);
    printf("test_module: ok\n");
    return 0;
}

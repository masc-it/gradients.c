/*
 * Intentionally empty: the custom powlu_split_linear backward-x12 Metal kernel
 * was removed from dispatch because the generic linear + powlu_split backward
 * path is faster for the GPT LM batch-96 training shape.
 */

/*
 * Intentionally empty: powlu_split_linear no longer has a custom Metal
 * backward-x12 entry point. The autograd implementation uses the faster
 * generic path: linear backward for dact followed by powlu_split backward.
 */

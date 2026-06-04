#ifndef GD_METAL_ABI_H
#define GD_METAL_ABI_H

/* Shared host/Metal scalar aliases for kernel ABI structs. */

#ifdef __METAL_VERSION__

typedef ulong gd_metal_u64;
typedef uint gd_metal_u32;

#else

#include <stddef.h>
#include <stdint.h>

typedef uint64_t gd_metal_u64;
typedef uint32_t gd_metal_u32;

#endif /* __METAL_VERSION__ */

#endif /* GD_METAL_ABI_H */

#include "metal_probe.h"

#import <Foundation/Foundation.h>

/* Compiles as Objective-C with ARC; proves the macOS toolchain wiring.
 * Replaced by the real Metal backend (device/queue/buffers/pipelines) later. */
int _gd_metal_toolchain_probe(void)
{
    @autoreleasepool {
        NSString *name = @"gradients.metal";
        return (int)[name length];
    }
}

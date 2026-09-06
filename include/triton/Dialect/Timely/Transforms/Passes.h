#ifndef TRITON_DIALECT_TIMELY_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TIMELY_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Timely/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace timely {

#define GEN_PASS_DECL
#include "triton/Dialect/Timely/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Timely/Transforms/Passes.h.inc"

} // namespace timely
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TIMELY_TRANSFORMS_PASSES_H_

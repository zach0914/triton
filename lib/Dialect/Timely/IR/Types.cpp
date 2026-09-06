#include "triton/Dialect/Timely/IR/Types.h"

#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/Timely/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::timely;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Timely/IR/Types.cpp.inc"

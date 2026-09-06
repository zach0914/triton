#include "triton/Dialect/Timely/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::triton::timely;

#include "triton/Dialect/Timely/IR/Dialect.cpp.inc"

void TimelyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Timely/IR/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Timely/IR/Ops.cpp.inc"
      >();
}

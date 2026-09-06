#ifndef TRITON_DIALECT_TIMELY_IR_DIALECT_H_
#define TRITON_DIALECT_TIMELY_IR_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Timely/IR/Dialect.h.inc"
#include "triton/Dialect/Timely/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/Timely/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TIMELY_IR_DIALECT_H_

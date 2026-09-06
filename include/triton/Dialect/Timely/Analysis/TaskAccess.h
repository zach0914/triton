#ifndef TRITON_DIALECT_TIMELY_ANALYSIS_TASKACCESS_H_
#define TRITON_DIALECT_TIMELY_ANALYSIS_TASKACCESS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Timely/IR/Dialect.h"

#include <string>

namespace mlir {
namespace triton {
namespace timely {

enum class TaskBufferFootprint { None, Exact, Unknown };

/// Runtime storage identity carried by a typed TM value. `region == "*"`
/// denotes the complete root binding. Other region strings are canonical view
/// tokens produced by the Timely frontend.
struct TaskValueProvenance {
  std::string root;
  std::string region;
  std::string lookupKey;

  bool isKnown() const {
    return !root.empty() && !region.empty() && !lookupKey.empty();
  }
};

/// Memory effects attributed to one outlined callee argument.
struct TaskFormalAccess {
  unsigned argumentIndex = 0;
  bool reads = false;
  bool writes = false;
  TaskBufferFootprint footprint = TaskBufferFootprint::None;
  SmallVector<unsigned> aliasArguments;
};

/// A formal access projected through a tm.task call onto its typed actual.
struct TaskActualAccess {
  unsigned argumentIndex = 0;
  Value value;
  bool reads = false;
  bool writes = false;
  TaskBufferFootprint footprint = TaskBufferFootprint::None;
  TaskValueProvenance provenance;
};

struct TaskAccessSummary {
  /// Explicit cross-task annotations. These only supplement effects whose
  /// storage provenance cannot be proven from the body and typed actuals.
  SmallVector<std::string> reads;
  SmallVector<std::string> writes;
  SmallVector<TaskFormalAccess> formalAccesses;
  SmallVector<TaskActualAccess> actualAccesses;
  unsigned bodyReadEffects = 0;
  unsigned bodyWriteEffects = 0;
  unsigned exactBufferRegions = 0;
  unsigned unknownBufferRegions = 0;
  unsigned aliasRoots = 0;
  unsigned unresolvedReadEffects = 0;
  unsigned unresolvedWriteEffects = 0;
};

/// Summarize a TM operation's cross-task access contract. A tm.task requires an
/// outlined Triton callee in the same module; its effects are projected from
/// formal arguments to typed actual storage provenance.
FailureOr<TaskAccessSummary> getTaskAccessSummary(Operation *operation);

} // namespace timely
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TIMELY_ANALYSIS_TASKACCESS_H_

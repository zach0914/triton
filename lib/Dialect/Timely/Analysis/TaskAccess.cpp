#include "triton/Dialect/Timely/Analysis/TaskAccess.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

#include <algorithm>

namespace mlir {
namespace triton {
namespace timely {

namespace {

struct BodyAccess {
  Value value;
  bool reads = false;
  bool writes = false;
  SmallVector<unsigned> formalArguments;
  TaskBufferFootprint footprint = TaskBufferFootprint::None;
};

bool isMemoryHandleType(Type type) {
  if (isa<triton::PointerType, triton::TensorDescType,
          triton::gpu::MemDescType>(type))
    return true;
  if (auto shaped = dyn_cast<ShapedType>(type))
    return isMemoryHandleType(shaped.getElementType());
  return false;
}

void collectFormalArguments(Value value, triton::FuncOp callee,
                            llvm::SmallBitVector &arguments,
                            llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;

  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    if (blockArgument.getOwner() == &callee.getBody().front() &&
        isMemoryHandleType(blockArgument.getType()))
      arguments.set(blockArgument.getArgNumber());
    return;
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  // Pointer arithmetic, broadcasts, descriptor construction and casts all
  // preserve provenance through one or more operands. Following every operand
  // is conservative: only pointer-like entry arguments are retained as roots.
  for (Value operand : definition->getOperands())
    collectFormalArguments(operand, callee, arguments, visited);
}

TaskBufferFootprint mergeFootprints(TaskBufferFootprint lhs,
                                    TaskBufferFootprint rhs) {
  if (lhs == TaskBufferFootprint::Unknown ||
      rhs == TaskBufferFootprint::Unknown)
    return TaskBufferFootprint::Unknown;
  if (lhs == TaskBufferFootprint::Exact || rhs == TaskBufferFootprint::Exact)
    return TaskBufferFootprint::Exact;
  return TaskBufferFootprint::None;
}

TaskValueProvenance getProvenanceAttrs(Operation *operation) {
  TaskValueProvenance provenance;
  if (!operation)
    return provenance;
  auto root = operation->getAttrOfType<StringAttr>("root");
  auto region = operation->getAttrOfType<StringAttr>("region");
  auto lookupKey = operation->getAttrOfType<StringAttr>("lookup_key");
  if (root && region && lookupKey) {
    provenance.root = root.getValue().str();
    provenance.region = region.getValue().str();
    provenance.lookupKey = lookupKey.getValue().str();
  }
  return provenance;
}

TaskValueProvenance getValueProvenance(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return {};

  if (definition->getName().getStringRef() == "tm.binding") {
    TaskValueProvenance provenance = getProvenanceAttrs(definition);
    if (provenance.isKnown())
      return provenance;
  }

  // A collective's data result denotes its destination rather than its source,
  // so the emitter records destination provenance on the operation itself.
  if (isa<AllGatherShardOp>(definition)) {
    TaskValueProvenance provenance = getProvenanceAttrs(definition);
    if (provenance.isKnown())
      return provenance;
  }

  return {};
}

bool annotationMatches(StringRef annotation,
                       const TaskValueProvenance &provenance) {
  return provenance.isKnown() &&
         (annotation == provenance.lookupKey || annotation == provenance.root);
}

LogicalResult validateAnnotations(TaskOp task, TaskAccessSummary &summary) {
  auto validateDirection = [&](ArrayRef<std::string> annotations,
                               bool checkReads) -> LogicalResult {
    unsigned unresolved = checkReads ? summary.unresolvedReadEffects
                                     : summary.unresolvedWriteEffects;
    if (unresolved && annotations.empty()) {
      return task.emitOpError()
             << "callee has " << (checkReads ? "read" : "write")
             << " effects that cannot be mapped to a formal argument; add a "
             << (checkReads ? "reads" : "writes") << " annotation";
    }
    if (unresolved)
      return success();

    for (const std::string &annotation : annotations) {
      bool matched = llvm::any_of(
          summary.actualAccesses, [&](const TaskActualAccess &access) {
            bool hasEffect = checkReads ? access.reads : access.writes;
            return hasEffect && annotationMatches(annotation, access.provenance);
          });
      if (!matched)
        return task.emitOpError()
               << (checkReads ? "reads" : "writes") << " annotation '"
               << annotation
               << "' contradicts the body-backed actual operand accesses";
    }
    return success();
  };

  if (failed(validateDirection(summary.reads, true)) ||
      failed(validateDirection(summary.writes, false)))
    return failure();
  return success();
}

} // namespace

FailureOr<TaskAccessSummary> getTaskAccessSummary(Operation *operation) {
  TaskAccessSummary summary;
  if (ArrayAttr reads = operation->getAttrOfType<ArrayAttr>("reads"))
    for (Attribute value : reads)
      summary.reads.push_back(cast<StringAttr>(value).getValue().str());
  if (ArrayAttr writes = operation->getAttrOfType<ArrayAttr>("writes"))
    for (Attribute value : writes)
      summary.writes.push_back(cast<StringAttr>(value).getValue().str());

  auto task = dyn_cast<TaskOp>(operation);
  if (!task)
    return summary;
  auto callee = SymbolTable::lookupNearestSymbolFrom<triton::FuncOp>(
      task, task.getCalleeAttr());
  if (!callee) {
    task.emitOpError("cannot analyze accesses because callee '")
        << task.getCalleeAttr().getValue()
        << "' does not resolve to a tt.func";
    return failure();
  }
  if (callee.isExternal()) {
    task.emitOpError("cannot analyze accesses because callee '")
        << task.getCalleeAttr().getValue() << "' has no body";
    return failure();
  }

  SmallVector<BodyAccess> bodyAccesses;
  callee.walk([&](Operation *op) {
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const MemoryEffects::EffectInstance &instance : instances) {
      bool reads = isa<MemoryEffects::Read>(instance.getEffect());
      bool writes = isa<MemoryEffects::Write>(instance.getEffect());
      if (!reads && !writes)
        continue;
      summary.bodyReadEffects += reads;
      summary.bodyWriteEffects += writes;

      BodyAccess access{instance.getValue(), reads, writes};
      if (access.value) {
        llvm::SmallBitVector formalArguments(callee.getNumArguments());
        llvm::DenseSet<Value> visited;
        collectFormalArguments(access.value, callee, formalArguments, visited);
        for (int argument = formalArguments.find_first(); argument >= 0;
             argument = formalArguments.find_next(argument))
          access.formalArguments.push_back(argument);
      }
      if (access.formalArguments.empty()) {
        summary.unresolvedReadEffects += reads;
        summary.unresolvedWriteEffects += writes;
      }
      bodyAccesses.push_back(std::move(access));
    }
  });

  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  auto *aliases = solver->load<SharedMemoryAliasAnalysis>();
  auto *regions = solver->load<BufferRegionAnalysis>();
  if (failed(solver->initializeAndRun(callee))) {
    task.emitOpError("failed to analyze the outlined task's buffer regions");
    return failure();
  }

  llvm::DenseSet<Value> localAliasRoots;
  llvm::SmallBitVector formalAliasRoots(callee.getNumArguments());
  for (BodyAccess &access : bodyAccesses) {
    const BufferRegionFootprint *footprint =
        access.value ? regions->getFootprint(access.value, callee) : nullptr;
    access.footprint = footprint ? TaskBufferFootprint::Exact
                                 : TaskBufferFootprint::Unknown;
    if (footprint)
      ++summary.exactBufferRegions;
    else
      ++summary.unknownBufferRegions;

    for (unsigned argument : access.formalArguments)
      formalAliasRoots.set(argument);
    if (!access.value || !isa<triton::gpu::MemDescType>(access.value.getType()))
      continue;
    auto *lattice = aliases->getLatticeElement(access.value);
    if (!lattice)
      continue;
    for (Value root : lattice->getValue().getAllocs())
      localAliasRoots.insert(root);
  }
  summary.aliasRoots = formalAliasRoots.count() + localAliasRoots.size();

  llvm::DenseMap<unsigned, unsigned> formalToAccess;
  for (const BodyAccess &bodyAccess : bodyAccesses) {
    for (unsigned argument : bodyAccess.formalArguments) {
      auto [it, inserted] =
          formalToAccess.try_emplace(argument, summary.formalAccesses.size());
      if (inserted) {
        TaskFormalAccess formal;
        formal.argumentIndex = argument;
        summary.formalAccesses.push_back(std::move(formal));
      }
      TaskFormalAccess &formal = summary.formalAccesses[it->second];
      formal.reads |= bodyAccess.reads;
      formal.writes |= bodyAccess.writes;
      formal.footprint = mergeFootprints(formal.footprint,
                                         bodyAccess.footprint);
      for (unsigned alias : bodyAccess.formalArguments)
        if (!llvm::is_contained(formal.aliasArguments, alias))
          formal.aliasArguments.push_back(alias);
    }
  }
  llvm::sort(summary.formalAccesses,
             [](const TaskFormalAccess &lhs, const TaskFormalAccess &rhs) {
               return lhs.argumentIndex < rhs.argumentIndex;
             });

  ValueRange actuals = task.getInputs();
  for (const TaskFormalAccess &formal : summary.formalAccesses) {
    if (formal.argumentIndex >= actuals.size()) {
      task.emitOpError("body access references missing actual operand #")
          << formal.argumentIndex;
      return failure();
    }
    TaskActualAccess actual;
    actual.argumentIndex = formal.argumentIndex;
    actual.value = actuals[formal.argumentIndex];
    actual.reads = formal.reads;
    actual.writes = formal.writes;
    actual.footprint = formal.footprint;
    actual.provenance = getValueProvenance(actual.value);
    if (!actual.provenance.isKnown()) {
      task.emitOpError("body-accessed actual operand #")
          << formal.argumentIndex
          << " has no typed tm.binding or collective-result provenance";
      return failure();
    }
    summary.actualAccesses.push_back(std::move(actual));
  }

  if (failed(validateAnnotations(task, summary)))
    return failure();
  return summary;
}

} // namespace timely
} // namespace triton
} // namespace mlir

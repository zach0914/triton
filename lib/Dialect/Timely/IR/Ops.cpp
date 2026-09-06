#include "triton/Dialect/Timely/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::timely;

#define GET_OP_CLASSES
#include "triton/Dialect/Timely/IR/Ops.cpp.inc"

namespace {

LogicalResult verifyStringArray(Operation *op, ArrayAttr values,
                                StringRef name) {
  for (Attribute value : values)
    if (!isa<StringAttr>(value))
      return op->emitOpError() << "requires '" << name
                               << "' entries to be strings";
  return success();
}

LogicalResult verifyResources(Operation *op, StringRef id,
                              StringRef resourceClass, int64_t sms,
                              int64_t threads, int64_t warps,
                              int64_t sharedMemoryBytes) {
  if (id.empty())
    return op->emitOpError("requires a non-empty task id");
  if (resourceClass != "communication" && resourceClass != "compute")
    return op->emitOpError(
        "resource_class must be 'communication' or 'compute'");
  if (sms <= 0 || threads <= 0 || warps <= 0 || sharedMemoryBytes < 0)
    return op->emitOpError(
        "requires positive sms/threads/warps and non-negative shared memory");
  return success();
}

LogicalResult verifyTaskAnnotations(Operation *op, StringRef id,
                                    ArrayAttr reads, ArrayAttr writes,
                                    ArrayAttr dependsOn, bool unknownAccess) {
  if (failed(verifyStringArray(op, reads, "reads")) ||
      failed(verifyStringArray(op, writes, "writes")) ||
      failed(verifyStringArray(op, dependsOn, "depends_on")))
    return failure();
  if (unknownAccess)
    return op->emitOpError(
        "unknown accesses are unsupported; add reads/writes annotations");
  for (Attribute dependency : dependsOn)
    if (cast<StringAttr>(dependency).getValue() == id)
      return op->emitOpError("cannot depend on itself");
  return success();
}

LogicalResult verifyDictionaryArray(Operation *op, ArrayAttr values,
                                    StringRef name) {
  for (Attribute value : values)
    if (!isa<DictionaryAttr>(value))
      return op->emitOpError() << "requires '" << name
                               << "' entries to be dictionaries";
  return success();
}

} // namespace

LogicalResult BindingOp::verify() {
  if (getRoot().empty())
    return emitOpError("requires a non-empty root");
  if (getRegion().empty())
    return emitOpError("requires a non-empty region");
  if (getLookupKey().empty())
    return emitOpError("requires a non-empty lookup_key");
  if (isa<TimeType, ConstType, EventType>(getResult().getType()))
    return emitOpError("cannot bind a Timely scheduling type");

  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  for (BindingOp binding : module.getOps<BindingOp>()) {
    if (binding == *this)
      break;
    if (binding.getLookupKey() == getLookupKey())
      return emitOpError("requires a unique lookup_key; '")
             << getLookupKey() << "' was already declared";
  }
  return success();
}

LogicalResult DomainOp::verify() {
  if (getLower().empty())
    return emitOpError("requires at least one domain dimension");
  if (getLower().size() != getUpper().size())
    return emitOpError("requires matching lower and upper ranks");
  for (auto [lower, upper] : llvm::zip(getLower(), getUpper()))
    if (lower >= upper)
      return emitOpError("requires every lower bound to be smaller than upper");

  llvm::StringSet<> parameterNames;
  for (Attribute parameter : getParameters()) {
    auto dictionary = dyn_cast<DictionaryAttr>(parameter);
    if (!dictionary)
      return emitOpError("requires parameters to be dictionaries");
    auto name = dictionary.getAs<StringAttr>("name");
    auto value = dictionary.getAs<IntegerAttr>("value");
    if (!name || name.getValue().empty() || !value ||
        !value.getType().isInteger(64))
      return emitOpError(
          "requires each parameter to contain a name and i64 value");
    if (!parameterNames.insert(name.getValue()).second)
      return emitOpError("requires unique scheduling parameter names");
  }
  return success();
}

LogicalResult AllGatherShardOp::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("source and gathered result types must match");
  if (getResourceClass() != "communication")
    return emitOpError("requires the communication resource class");
  if (getWrites().empty())
    return emitOpError("requires a destination in writes");
  if (failed(verifyResources(*this, getId(), getResourceClass(), getSms(),
                             getThreads(), getWarps(),
                             getSharedMemoryBytes())))
    return failure();
  return verifyTaskAnnotations(*this, getId(), getReads(), getWrites(),
                               getDependsOn(), getUnknownAccess());
}

LogicalResult TaskOp::verify() {
  if (getSpecialization().empty())
    return emitOpError("requires a non-empty specialization identity");
  if (getResourceClass() != "compute")
    return emitOpError("requires the compute resource class");
  if (failed(verifyResources(*this, getId(), getResourceClass(), getSms(),
                             getThreads(), getWarps(),
                             getSharedMemoryBytes())))
    return failure();
  return verifyTaskAnnotations(*this, getId(), getReads(), getWrites(),
                               getDependsOn(), getUnknownAccess());
}

LogicalResult
TaskOp::verifySymbolUses(SymbolTableCollection &symbolTableCollection) {
  Operation *symbol = symbolTableCollection.lookupNearestSymbolFrom(
      *this, getCalleeAttr());
  StringRef taskId = getId();
  StringRef calleeName = getCalleeAttr().getValue();
  if (!symbol)
    return emitOpError() << "task '" << taskId << "' callee '@" << calleeName
                         << "' does not resolve to a symbol";

  auto callee = dyn_cast<triton::FuncOp>(symbol);
  if (!callee)
    return emitOpError() << "task '" << taskId << "' callee '@" << calleeName
                         << "' must resolve to a tt.func, but resolved to '"
                         << symbol->getName() << "'";
  if (callee.isExternal())
    return emitOpError() << "task '" << taskId << "' callee '@" << calleeName
                         << "' must have a body";

  FunctionType calleeType = callee.getFunctionType();
  if (calleeType.getNumResults() != 0)
    return emitOpError() << "task '" << taskId << "' callee '@" << calleeName
                         << "' must return void, but returns "
                         << calleeType.getResults();

  if (calleeType.getNumInputs() != getInputs().size())
    return emitOpError() << "task '" << taskId << "' callee '@" << calleeName
                         << "' expects " << calleeType.getNumInputs()
                         << " dynamic arguments, but task provides "
                         << getInputs().size();

  for (auto [index, actual] : llvm::enumerate(getInputs())) {
    Type expected = calleeType.getInput(index);
    if (actual.getType() != expected)
      return emitOpError()
             << "task '" << taskId << "' callee '@" << calleeName
             << "' argument " << index << " type mismatch: expected "
             << expected << ", but got " << actual.getType();

    if (!isa<triton::PointerType>(actual.getType()))
      continue;
    Value provenance = actual;
    while (auto allGather =
               provenance.getDefiningOp<AllGatherShardOp>())
      provenance = allGather.getSource();
    if (!provenance.getDefiningOp<BindingOp>())
      return emitOpError()
             << "task '" << taskId << "' callee '@" << calleeName
             << "' pointer argument " << index
             << " must originate from a tm.binding";
  }
  return success();
}

LogicalResult GraphOp::verify() {
  if (failed(verifyDictionaryArray(*this, getNodes(), "nodes")) ||
      failed(verifyDictionaryArray(*this, getTimeOrder(), "time_order")) ||
      failed(verifyDictionaryArray(*this, getDataDeps(), "data_deps")))
    return failure();
  return success();
}

LogicalResult PlanOp::verify() {
  if (failed(verifyDictionaryArray(*this, getNodes(), "nodes")) ||
      failed(verifyDictionaryArray(*this, getIssueLayers(), "issue_layers")) ||
      failed(verifyDictionaryArray(*this, getTimeOrder(), "time_order")) ||
      failed(verifyDictionaryArray(*this, getDataDeps(), "data_deps")) ||
      failed(verifyDictionaryArray(*this, getResourceOrder(),
                                   "resource_order")) ||
      failed(verifyDictionaryArray(*this, getSynchronizations(),
                                   "synchronizations")))
    return failure();
  return success();
}

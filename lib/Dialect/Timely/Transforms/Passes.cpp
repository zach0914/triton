#include "triton/Dialect/Timely/Transforms/Passes.h"
#include "triton/Dialect/Timely/Analysis/TaskAccess.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace triton {
namespace timely {

#define GEN_PASS_DEF_TIMELYNORMALIZEISSUETIME
#define GEN_PASS_DEF_TIMELYBUILDDEPENDENCYGRAPH
#define GEN_PASS_DEF_TIMELYPLANRESOURCES
#define GEN_PASS_DEF_TIMELYMATERIALIZESYNCHRONIZATION
#include "triton/Dialect/Timely/Transforms/Passes.h.inc"

namespace {

struct NodeInfo {
  Operation *op;
  std::string id;
  std::string kind;
  std::string callee;
  std::string specialization;
  int64_t rawTime = 0;
  int64_t rank = 0;
};

struct Edge {
  unsigned from;
  unsigned to;
  std::string kind;

  bool operator<(const Edge &other) const {
    return std::tie(from, to, kind) <
           std::tie(other.from, other.to, other.kind);
  }
};

SmallVector<Operation *> getTaskOps(ModuleOp module) {
  SmallVector<Operation *> result;
  module.walk([&](Operation *op) {
    if (isa<AllGatherShardOp, TaskOp>(op))
      result.push_back(op);
  });
  return result;
}

StringRef getTaskId(Operation *op) {
  return op->getAttrOfType<StringAttr>("id").getValue();
}

Value getTaskTime(Operation *op) {
  if (auto communication = dyn_cast<AllGatherShardOp>(op))
    return communication.getTime();
  return cast<TaskOp>(op).getTime();
}

FailureOr<int64_t> resolveConst(Value value) {
  auto constant = value.getDefiningOp<ConstConstantOp>();
  if (!constant)
    return failure();
  return constant.getValueAttr().getInt();
}

FailureOr<int64_t> resolveTime(Value value) {
  if (auto constant = value.getDefiningOp<TimeConstantOp>())
    return constant.getValueAttr().getInt();
  auto add = value.getDefiningOp<TimeAddOp>();
  if (!add)
    return failure();
  FailureOr<int64_t> base = resolveTime(add.getBase());
  FailureOr<int64_t> offset = resolveConst(add.getOffset());
  if (failed(base) || failed(offset))
    return failure();
  return *base + *offset;
}

SmallVector<std::string> getStringArray(Operation *op, StringRef name) {
  SmallVector<std::string> result;
  for (Attribute value : op->getAttrOfType<ArrayAttr>(name))
    result.push_back(cast<StringAttr>(value).getValue().str());
  return result;
}

DictionaryAttr makeNodeAttr(OpBuilder &builder, const NodeInfo &node) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("id", builder.getStringAttr(node.id)),
      builder.getNamedAttr("kind", builder.getStringAttr(node.kind)),
      builder.getNamedAttr("callee", builder.getStringAttr(node.callee)),
      builder.getNamedAttr("specialization",
                           builder.getStringAttr(node.specialization)),
      builder.getNamedAttr("raw_time", builder.getI64IntegerAttr(node.rawTime)),
      builder.getNamedAttr("issue_rank", builder.getI64IntegerAttr(node.rank)),
  });
}

DictionaryAttr makeEdgeAttr(OpBuilder &builder, const Edge &edge,
                            ArrayRef<NodeInfo> nodes) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("from", builder.getStringAttr(nodes[edge.from].id)),
      builder.getNamedAttr("to", builder.getStringAttr(nodes[edge.to].id)),
      builder.getNamedAttr("kind", builder.getStringAttr(edge.kind)),
  });
}

SmallVector<NodeInfo> collectNodes(ModuleOp module, bool requireRank) {
  SmallVector<NodeInfo> result;
  for (Operation *op : getTaskOps(module)) {
    FailureOr<int64_t> rawTime = resolveTime(getTaskTime(op));
    if (failed(rawTime))
      continue;
    auto rank = op->getAttrOfType<IntegerAttr>("tm.issue_rank");
    auto task = dyn_cast<TaskOp>(op);
    auto specialization = op->getAttrOfType<StringAttr>("specialization");
    result.push_back({op, getTaskId(op).str(),
                      isa<AllGatherShardOp>(op) ? "communication" : "compute",
                      task ? task.getCalleeAttr().getValue().str() : "",
                      specialization ? specialization.getValue().str() : "",
                      *rawTime, rank ? rank.getInt() : 0});
    if (requireRank && !rank)
      result.pop_back();
  }
  return result;
}

StringRef getFootprintName(TaskBufferFootprint footprint) {
  switch (footprint) {
  case TaskBufferFootprint::None:
    return "none";
  case TaskBufferFootprint::Exact:
    return "exact";
  case TaskBufferFootprint::Unknown:
    return "unknown";
  }
  llvm_unreachable("unhandled task buffer footprint");
}

ArrayAttr makeFormalAccessAttrs(OpBuilder &builder,
                                const TaskAccessSummary &summary) {
  SmallVector<Attribute> attributes;
  for (const TaskFormalAccess &access : summary.formalAccesses) {
    SmallVector<int64_t> aliases(access.aliasArguments.begin(),
                                 access.aliasArguments.end());
    attributes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("argument",
                             builder.getI64IntegerAttr(access.argumentIndex)),
        builder.getNamedAttr("reads", builder.getBoolAttr(access.reads)),
        builder.getNamedAttr("writes", builder.getBoolAttr(access.writes)),
        builder.getNamedAttr("footprint",
                             builder.getStringAttr(
                                 getFootprintName(access.footprint))),
        builder.getNamedAttr("alias_arguments",
                             builder.getDenseI64ArrayAttr(aliases)),
    }));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr makeActualAccessAttrs(OpBuilder &builder,
                                const TaskAccessSummary &summary) {
  SmallVector<Attribute> attributes;
  for (const TaskActualAccess &access : summary.actualAccesses) {
    attributes.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("argument",
                             builder.getI64IntegerAttr(access.argumentIndex)),
        builder.getNamedAttr("reads", builder.getBoolAttr(access.reads)),
        builder.getNamedAttr("writes", builder.getBoolAttr(access.writes)),
        builder.getNamedAttr("footprint",
                             builder.getStringAttr(
                                 getFootprintName(access.footprint))),
        builder.getNamedAttr("root",
                             builder.getStringAttr(access.provenance.root)),
        builder.getNamedAttr("region",
                             builder.getStringAttr(access.provenance.region)),
        builder.getNamedAttr(
            "lookup_key",
            builder.getStringAttr(access.provenance.lookupKey)),
    }));
  }
  return builder.getArrayAttr(attributes);
}

enum class StorageOverlap { No, Yes, Unknown };

bool isCanonicalShardRegion(StringRef region) {
  if (!region.starts_with('[') || !region.ends_with(']') || region.size() < 3)
    return false;
  SmallVector<StringRef> coordinates;
  region.drop_front().drop_back().split(coordinates, ',');
  if (coordinates.empty())
    return false;
  for (StringRef coordinate : coordinates) {
    uint64_t value;
    if (coordinate.empty() || coordinate.getAsInteger(10, value))
      return false;
  }
  return true;
}

StorageOverlap compareStorage(const TaskValueProvenance &lhs,
                              const TaskValueProvenance &rhs) {
  if (!lhs.isKnown() || !rhs.isKnown())
    return StorageOverlap::Unknown;
  if (lhs.root != rhs.root)
    return StorageOverlap::No;
  auto isWhole = [](StringRef region) {
    return region == "*" || region == "whole";
  };
  if (isWhole(lhs.region) || isWhole(rhs.region) ||
      lhs.region == rhs.region)
    return StorageOverlap::Yes;
  if (isCanonicalShardRegion(lhs.region) &&
      isCanonicalShardRegion(rhs.region))
    return StorageOverlap::No;
  return StorageOverlap::Unknown;
}

bool annotationsOverlap(ArrayRef<std::string> lhs,
                        ArrayRef<std::string> rhs) {
  return llvm::any_of(lhs, [&](const std::string &left) {
    return llvm::is_contained(rhs, left);
  });
}

void eraseOps(ModuleOp module, StringRef operationName) {
  SmallVector<Operation *> stale;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == operationName)
      stale.push_back(op);
  });
  for (Operation *op : stale)
    op->erase();
}

class NormalizeIssueTimePass
    : public impl::TimelyNormalizeIssueTimeBase<NormalizeIssueTimePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> taskOps = getTaskOps(module);
    std::map<int64_t, int64_t> ranks;
    llvm::StringSet<> ids;

    for (Operation *op : taskOps) {
      if (!ids.insert(getTaskId(op)).second) {
        op->emitError("duplicate Timely task id '") << getTaskId(op) << "'";
        return signalPassFailure();
      }
      FailureOr<int64_t> time = resolveTime(getTaskTime(op));
      if (failed(time)) {
        op->emitError(
            "logical issue time must be fully specialized from tm constants");
        return signalPassFailure();
      }
      ranks.emplace(*time, 0);
    }

    int64_t nextRank = 0;
    for (auto &[time, rank] : ranks)
      rank = nextRank++;
    OpBuilder builder(module.getContext());
    for (Operation *op : taskOps) {
      int64_t time = *resolveTime(getTaskTime(op));
      op->setAttr("tm.raw_issue_time", builder.getI64IntegerAttr(time));
      op->setAttr("tm.issue_rank", builder.getI64IntegerAttr(ranks[time]));
    }
  }
};

class BuildDependencyGraphPass
    : public impl::TimelyBuildDependencyGraphBase<BuildDependencyGraphPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<NodeInfo> nodes = collectNodes(module, true);
    SmallVector<Operation *> taskOps = getTaskOps(module);
    if (nodes.size() != taskOps.size()) {
      module.emitError(
          "timely-build-dependency-graph requires normalized issue times");
      return signalPassFailure();
    }

    llvm::StringMap<unsigned> idToIndex;
    llvm::DenseMap<Operation *, unsigned> opToIndex;
    for (auto [index, node] : llvm::enumerate(nodes)) {
      idToIndex[node.id] = index;
      opToIndex[node.op] = index;
    }

    SmallVector<TaskAccessSummary, 0> summaries;
    OpBuilder summaryBuilder(module.getContext());
    for (const NodeInfo &node : nodes) {
      FailureOr<TaskAccessSummary> summary = getTaskAccessSummary(node.op);
      if (failed(summary))
        return signalPassFailure();
      node.op->setAttr("tm.body_read_effects",
                       summaryBuilder.getI64IntegerAttr(summary->bodyReadEffects));
      node.op->setAttr(
          "tm.body_write_effects",
          summaryBuilder.getI64IntegerAttr(summary->bodyWriteEffects));
      node.op->setAttr(
          "tm.exact_buffer_regions",
          summaryBuilder.getI64IntegerAttr(summary->exactBufferRegions));
      node.op->setAttr(
          "tm.unknown_buffer_regions",
          summaryBuilder.getI64IntegerAttr(summary->unknownBufferRegions));
      node.op->setAttr("tm.alias_roots",
                       summaryBuilder.getI64IntegerAttr(summary->aliasRoots));
      node.op->setAttr("tm.unresolved_read_effects",
                       summaryBuilder.getI64IntegerAttr(
                           summary->unresolvedReadEffects));
      node.op->setAttr("tm.unresolved_write_effects",
                       summaryBuilder.getI64IntegerAttr(
                           summary->unresolvedWriteEffects));
      node.op->setAttr("tm.formal_accesses",
                       makeFormalAccessAttrs(summaryBuilder, *summary));
      node.op->setAttr("tm.actual_accesses",
                       makeActualAccessAttrs(summaryBuilder, *summary));
      summaries.push_back(std::move(*summary));
    }

    std::set<Edge> dataEdges;
    auto addDataEdge = [&](unsigned from, unsigned to, StringRef kind) {
      dataEdges.insert({from, to, kind.str()});
    };

    for (auto [consumerIndex, node] : llvm::enumerate(nodes)) {
      for (Value operand : node.op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        auto producerIt = opToIndex.find(producer);
        if (producerIt == opToIndex.end())
          continue;
        addDataEdge(producerIt->second, consumerIndex,
                    isa<EventType>(operand.getType()) ? "EVENT" : "SSA");
      }

      for (const std::string &dependency :
           getStringArray(node.op, "depends_on")) {
        auto producer = idToIndex.find(dependency);
        if (producer == idToIndex.end()) {
          node.op->emitOpError("depends on missing producer '")
              << dependency << "'";
          return signalPassFailure();
        }
        addDataEdge(producer->second, consumerIndex, "EXPLICIT");
      }
    }

    for (unsigned from = 0; from < nodes.size(); ++from) {
      ArrayRef<std::string> fromReads = summaries[from].reads;
      ArrayRef<std::string> fromWrites = summaries[from].writes;
      llvm::StringSet<> readSet;
      llvm::StringSet<> writeSet;
      for (const std::string &access : fromReads)
        readSet.insert(access);
      for (const std::string &access : fromWrites)
        writeSet.insert(access);
      for (unsigned to = from + 1; to < nodes.size(); ++to) {
        for (const std::string &access : summaries[to].reads)
          if (writeSet.contains(access))
            addDataEdge(from, to, "RAW");
        for (const std::string &access : summaries[to].writes) {
          if (readSet.contains(access))
            addDataEdge(from, to, "WAR");
          if (writeSet.contains(access))
            addDataEdge(from, to, "WAW");
        }

        auto addBodyHazard = [&](const TaskActualAccess &source,
                                 const TaskActualAccess &target,
                                 bool hasHazard, StringRef kind,
                                 ArrayRef<std::string> sourceAnnotations,
                                 ArrayRef<std::string> targetAnnotations)
            -> LogicalResult {
          if (!hasHazard)
            return success();
          switch (compareStorage(source.provenance, target.provenance)) {
          case StorageOverlap::No:
            return success();
          case StorageOverlap::Yes:
            addDataEdge(from, to, kind);
            return success();
          case StorageOverlap::Unknown:
            if (annotationsOverlap(sourceAnnotations, targetAnnotations)) {
              addDataEdge(from, to, kind);
              return success();
            }
            nodes[to].op->emitOpError("cannot determine whether body-backed ")
                << kind << " accesses overlap between task '" << nodes[from].id
                << "' argument #" << source.argumentIndex << " (root '"
                << source.provenance.root << "', region '"
                << source.provenance.region << "') and task '" << nodes[to].id
                << "' argument #" << target.argumentIndex << " (root '"
                << target.provenance.root << "', region '"
                << target.provenance.region
                << "'); add matching reads/writes region annotations";
            return failure();
          }
          llvm_unreachable("unhandled storage overlap");
        };

        for (const TaskActualAccess &source : summaries[from].actualAccesses) {
          for (const TaskActualAccess &target : summaries[to].actualAccesses) {
            if (failed(addBodyHazard(source, target,
                                     source.writes && target.reads, "RAW",
                                     fromWrites, summaries[to].reads)) ||
                failed(addBodyHazard(source, target,
                                     source.reads && target.writes, "WAR",
                                     fromReads, summaries[to].writes)) ||
                failed(addBodyHazard(source, target,
                                     source.writes && target.writes, "WAW",
                                     fromWrites, summaries[to].writes)))
              return signalPassFailure();
          }
        }
      }
    }

    SmallVector<SmallVector<unsigned>> successors(nodes.size());
    SmallVector<unsigned> indegree(nodes.size(), 0);
    std::set<std::pair<unsigned, unsigned>> graphEdges;
    for (const Edge &edge : dataEdges) {
      if (graphEdges.insert({edge.from, edge.to}).second) {
        successors[edge.from].push_back(edge.to);
        ++indegree[edge.to];
      }
      if (nodes[edge.from].rawTime > nodes[edge.to].rawTime) {
        nodes[edge.to].op->emitError("logical issue-time inversion: producer '")
            << nodes[edge.from].id << "' is issued after consumer '"
            << nodes[edge.to].id << "'";
        return signalPassFailure();
      }
    }

    std::queue<unsigned> ready;
    for (auto [index, degree] : llvm::enumerate(indegree))
      if (degree == 0)
        ready.push(index);
    unsigned visited = 0;
    while (!ready.empty()) {
      unsigned current = ready.front();
      ready.pop();
      ++visited;
      for (unsigned successor : successors[current])
        if (--indegree[successor] == 0)
          ready.push(successor);
    }
    if (visited != nodes.size()) {
      module.emitError("Timely data-dependency graph contains a cycle");
      return signalPassFailure();
    }

    std::set<Edge> timeEdges;
    std::map<int64_t, SmallVector<unsigned>> layers;
    for (auto [index, node] : llvm::enumerate(nodes))
      layers[node.rank].push_back(index);
    for (auto layer = layers.begin(); layer != layers.end(); ++layer) {
      auto next = std::next(layer);
      if (next == layers.end())
        break;
      for (unsigned from : layer->second)
        for (unsigned to : next->second)
          timeEdges.insert({from, to, "ISSUE"});
    }

    OpBuilder builder(module.getContext());
    SmallVector<Attribute> nodeAttrs;
    SmallVector<Attribute> timeAttrs;
    SmallVector<Attribute> dataAttrs;
    for (const NodeInfo &node : nodes)
      nodeAttrs.push_back(makeNodeAttr(builder, node));
    for (const Edge &edge : timeEdges)
      timeAttrs.push_back(makeEdgeAttr(builder, edge, nodes));
    for (const Edge &edge : dataEdges)
      dataAttrs.push_back(makeEdgeAttr(builder, edge, nodes));

    eraseOps(module, GraphOp::getOperationName());
    builder.setInsertionPointToEnd(module.getBody());
    OperationState state(module.getLoc(), GraphOp::getOperationName());
    state.addAttribute("nodes", builder.getArrayAttr(nodeAttrs));
    state.addAttribute("time_order", builder.getArrayAttr(timeAttrs));
    state.addAttribute("data_deps", builder.getArrayAttr(dataAttrs));
    builder.create(state);
  }
};

struct ResourceUsage {
  int64_t sms = 0;
  int64_t threads = 0;
  int64_t warps = 0;
  int64_t sharedMemoryBytes = 0;
};

ResourceUsage getResources(Operation *op) {
  return {
      op->getAttrOfType<IntegerAttr>("sms").getInt(),
      op->getAttrOfType<IntegerAttr>("threads").getInt(),
      op->getAttrOfType<IntegerAttr>("warps").getInt(),
      op->getAttrOfType<IntegerAttr>("shared_memory_bytes").getInt(),
  };
}

class PlanResourcesPass
    : public impl::TimelyPlanResourcesBase<PlanResourcesPass> {
public:
  using impl::TimelyPlanResourcesBase<PlanResourcesPass>::TimelyPlanResourcesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto graphs = module.getOps<GraphOp>();
    if (graphs.empty()) {
      module.emitError("timely-plan-resources requires a dependency graph");
      return signalPassFailure();
    }
    GraphOp graph = *graphs.begin();
    SmallVector<NodeInfo> nodes = collectNodes(module, true);
    llvm::StringMap<unsigned> idToIndex;
    for (auto [index, node] : llvm::enumerate(nodes))
      idToIndex[node.id] = index;

    SmallVector<SmallVector<unsigned>> successors(nodes.size());
    SmallVector<unsigned> indegree(nodes.size(), 0);
    std::set<std::pair<unsigned, unsigned>> dataPairs;
    for (Attribute attribute : graph.getDataDeps()) {
      DictionaryAttr edge = cast<DictionaryAttr>(attribute);
      unsigned from = idToIndex.lookup(cast<StringAttr>(edge.get("from")).getValue());
      unsigned to = idToIndex.lookup(cast<StringAttr>(edge.get("to")).getValue());
      if (dataPairs.insert({from, to}).second) {
        successors[from].push_back(to);
        ++indegree[to];
      }
    }
    std::set<unsigned> ready;
    for (auto [index, degree] : llvm::enumerate(indegree))
      if (degree == 0)
        ready.insert(index);
    SmallVector<unsigned> topologicalRank(nodes.size(), 0);
    unsigned nextTopologicalRank = 0;
    while (!ready.empty()) {
      unsigned current = *ready.begin();
      ready.erase(ready.begin());
      topologicalRank[current] = nextTopologicalRank++;
      for (unsigned successor : successors[current])
        if (--indegree[successor] == 0)
          ready.insert(successor);
    }
    if (nextTopologicalRank != nodes.size()) {
      module.emitError("timely-plan-resources requires an acyclic DataDep graph");
      return signalPassFailure();
    }

    std::map<std::pair<int64_t, std::string>, SmallVector<unsigned>> groups;
    for (auto [index, node] : llvm::enumerate(nodes))
      groups[{node.rank, node.kind}].push_back(index);

    std::set<Edge> resourceEdges;
    for (auto &[key, group] : groups) {
      llvm::sort(group, [&](unsigned lhs, unsigned rhs) {
        return topologicalRank[lhs] < topologicalRank[rhs];
      });
      std::optional<unsigned> previous;
      for (unsigned index : group) {
        ResourceUsage need = getResources(nodes[index].op);
        if (need.sms > maxSms || need.threads > maxThreads ||
            need.warps > maxWarps ||
            need.sharedMemoryBytes > maxSharedMemoryBytes) {
          nodes[index].op->emitOpError(
              "resource request exceeds target capacity");
          return signalPassFailure();
        }
        // TM deliberately uses a conservative placeholder until target-specific
        // occupancy and placement are available at a lower compiler level.
        if (previous)
          resourceEdges.insert({*previous, index, "MVP_SERIAL"});
        previous = index;
      }
    }

    OpBuilder builder(module.getContext());
    SmallVector<Attribute> layerAttrs;
    std::map<int64_t, SmallVector<const NodeInfo *>> layers;
    for (const NodeInfo &node : nodes)
      layers[node.rank].push_back(&node);
    for (auto &[rank, tasks] : layers) {
      llvm::sort(tasks, [](const NodeInfo *lhs, const NodeInfo *rhs) {
        return std::tie(lhs->kind, lhs->id) < std::tie(rhs->kind, rhs->id);
      });
      SmallVector<Attribute> taskAttrs;
      for (const NodeInfo *task : tasks)
        taskAttrs.push_back(builder.getStringAttr(task->id));
      layerAttrs.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("rank", builder.getI64IntegerAttr(rank)),
          builder.getNamedAttr("tasks", builder.getArrayAttr(taskAttrs)),
      }));
    }
    SmallVector<Attribute> resourceAttrs;
    for (const Edge &edge : resourceEdges)
      resourceAttrs.push_back(makeEdgeAttr(builder, edge, nodes));

    eraseOps(module, PlanOp::getOperationName());
    builder.setInsertionPointToEnd(module.getBody());
    OperationState state(module.getLoc(), PlanOp::getOperationName());
    state.addAttribute("nodes", graph.getNodesAttr());
    state.addAttribute("issue_layers", builder.getArrayAttr(layerAttrs));
    state.addAttribute("time_order", graph.getTimeOrderAttr());
    state.addAttribute("data_deps", graph.getDataDepsAttr());
    state.addAttribute("resource_order", builder.getArrayAttr(resourceAttrs));
    state.addAttribute("synchronizations", builder.getArrayAttr({}));
    builder.create(state);
  }
};

class MaterializeSynchronizationPass
    : public impl::TimelyMaterializeSynchronizationBase<
          MaterializeSynchronizationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto plans = module.getOps<PlanOp>();
    if (plans.empty()) {
      module.emitError("timely-materialize-synchronization requires a plan");
      return signalPassFailure();
    }
    PlanOp plan = *plans.begin();
    OpBuilder builder(module.getContext());
    SmallVector<Attribute> synchronizations;
    std::set<std::pair<std::string, std::string>> synchronizedPairs;
    for (Attribute attribute : plan.getDataDeps()) {
      DictionaryAttr edge = cast<DictionaryAttr>(attribute);
      std::string from = cast<StringAttr>(edge.get("from")).getValue().str();
      std::string to = cast<StringAttr>(edge.get("to")).getValue().str();
      if (!synchronizedPairs.insert({from, to}).second)
        continue;
      synchronizations.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("from", edge.get("from")),
          builder.getNamedAttr("to", edge.get("to")),
          builder.getNamedAttr("kind", builder.getStringAttr("completion_event")),
          builder.getNamedAttr("scope", builder.getStringAttr("cross_task")),
      }));
    }
    plan.setSynchronizationsAttr(builder.getArrayAttr(synchronizations));
  }
};

} // namespace
} // namespace timely
} // namespace triton
} // namespace mlir

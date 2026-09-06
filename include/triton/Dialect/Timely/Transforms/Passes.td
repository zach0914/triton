#ifndef TIMELY_PASSES
#define TIMELY_PASSES

include "mlir/Pass/PassBase.td"

def TimelyNormalizeIssueTime : Pass<"timely-normalize-issue-time", "mlir::ModuleOp"> {
  let summary = "Densify logical issue-time keys while preserving order";
  let dependentDialects = ["mlir::triton::timely::TimelyDialect"];
}

def TimelyBuildDependencyGraph : Pass<"timely-build-dependency-graph", "mlir::ModuleOp"> {
  let summary = "Build TimeOrder and DataDep edges and reject illegal schedules";
  let dependentDialects = ["mlir::triton::timely::TimelyDialect"];
}

def TimelyPlanResources : Pass<"timely-plan-resources", "mlir::ModuleOp"> {
  let summary = "Build issue layers and conservative MVP ResourceOrder edges";
  let dependentDialects = ["mlir::triton::timely::TimelyDialect"];
  let options = [
    Option<"maxSms", "max-sms", "int64_t", /*default*/"128",
           "Available SM slots per execution resource class">,
    Option<"maxThreads", "max-threads", "int64_t", /*default*/"2048",
           "Available thread slots per execution resource class">,
    Option<"maxWarps", "max-warps", "int64_t", /*default*/"64",
           "Available warp slots per execution resource class">,
    Option<"maxSharedMemoryBytes", "max-shared-memory-bytes", "int64_t",
           /*default*/"233472", "Available shared memory bytes">
  ];
}

def TimelyMaterializeSynchronization : Pass<"timely-materialize-synchronization", "mlir::ModuleOp"> {
  let summary = "Materialize completion waits only for DataDep edges";
  let dependentDialects = ["mlir::triton::timely::TimelyDialect"];
}

#endif // TIMELY_PASSES

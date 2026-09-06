#ifndef TIMELY_OPS
#define TIMELY_OPS

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Timely/IR/TimelyTypes.td"

def Timely_ConstConstantOp : Timely_Op<"const", [Pure]> {
  let summary = "Materialize a compile-time scheduling constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs Timely_ConstType:$result);
  let assemblyFormat = "$value attr-dict";
}

def Timely_TimeConstantOp : Timely_Op<"time.constant", [Pure]> {
  let summary = "Materialize a logical issue-time key";
  let arguments = (ins I64Attr:$value);
  let results = (outs Timely_TimeType:$result);
  let assemblyFormat = "$value attr-dict";
}

def Timely_TimeAddOp : Timely_Op<"time.add", [Pure]> {
  let summary = "Add a specialized scheduling constant to an issue-time key";
  let arguments = (ins Timely_TimeType:$base, Timely_ConstType:$offset);
  let results = (outs Timely_TimeType:$result);
  let assemblyFormat = "$base `,` $offset attr-dict `:` type($base)";
}

def Timely_EventNoneOp : Timely_Op<"event.none", [Pure]> {
  let summary = "Create the root event for a task without a predecessor";
  let results = (outs Timely_EventType:$result);
  let assemblyFormat = "attr-dict";
}

def Timely_BindingOp : Timely_Op<"binding", [Pure, HasParent<"ModuleOp">]> {
  let summary = "Declare a typed runtime value with storage provenance";
  let arguments = (ins StrAttr:$root,
                       StrAttr:$region,
                       StrAttr:$lookup_key);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Timely_DomainOp : Timely_Op<"domain", [Symbol]> {
  let summary = "Declare a finite, unordered, static task domain";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DenseI64ArrayAttr:$lower,
                       DenseI64ArrayAttr:$upper,
                       ArrayAttr:$parameters);
  let hasVerifier = 1;
}

def Timely_AllGatherShardOp : Timely_Op<"allgather_shard"> {
  let summary = "Issue an asynchronous shard communication task";
  let arguments = (ins AnyType:$source,
                       Timely_TimeType:$time,
                       Timely_EventType:$dependency,
                       StrAttr:$id,
                       ArrayAttr:$reads,
                       ArrayAttr:$writes,
                       ArrayAttr:$depends_on,
                       StrAttr:$resource_class,
                       I64Attr:$sms,
                       I64Attr:$threads,
                       I64Attr:$warps,
                       I64Attr:$shared_memory_bytes,
                       DefaultValuedAttr<BoolAttr, "false">:$unknown_access);
  let results = (outs AnyType:$result, Timely_EventType:$done);
  let hasVerifier = 1;
}

def Timely_TaskOp : Timely_Op<"task", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Issue an outlined Triton compute task at a logical time";
  let arguments = (ins Variadic<AnyType>:$inputs,
                       Timely_TimeType:$time,
                       Timely_EventType:$dependency,
                       FlatSymbolRefAttr:$callee,
                       StrAttr:$specialization,
                       StrAttr:$id,
                       ArrayAttr:$reads,
                       ArrayAttr:$writes,
                       ArrayAttr:$depends_on,
                       StrAttr:$resource_class,
                       I64Attr:$sms,
                       I64Attr:$threads,
                       I64Attr:$warps,
                       I64Attr:$shared_memory_bytes,
                       DefaultValuedAttr<BoolAttr, "false">:$unknown_access);
  let results = (outs Timely_EventType:$done);
  let hasVerifier = 1;
}

def Timely_GraphOp : Timely_Op<"graph"> {
  let summary = "An inspectable task dependency graph";
  let arguments = (ins ArrayAttr:$nodes,
                       ArrayAttr:$time_order,
                       ArrayAttr:$data_deps);
  let hasVerifier = 1;
}

def Timely_PlanOp : Timely_Op<"plan"> {
  let summary = "A target-aware issue, resource, and synchronization plan";
  let arguments = (ins ArrayAttr:$nodes,
                       ArrayAttr:$issue_layers,
                       ArrayAttr:$time_order,
                       ArrayAttr:$data_deps,
                       ArrayAttr:$resource_order,
                       ArrayAttr:$synchronizations);
  let hasVerifier = 1;
}

#endif // TIMELY_OPS

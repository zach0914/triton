// RUN: triton-opt %s | FileCheck %s

module {
  tt.func private @gemm(%gathered: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                        %output: !tt.ptr<f32>) {
    %lhs = tt.load %gathered : !tt.ptr<f32>
    %rhs = tt.load %b : !tt.ptr<f32>
    %value = arith.addf %lhs, %rhs : f32
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %lag = tm.const 1
  %t0 = tm.time.constant 0
  %t1 = tm.time.add %t0, %lag : !tm.time
  %root = tm.event.none
  %source = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %b = "tm.binding"() {
    root = "B", region = "*", lookup_key = "B"
  } : () -> !tt.ptr<f32>
  %output = "tm.binding"() {
    root = "C", region = "[0]", lookup_key = "C[0]"
  } : () -> !tt.ptr<f32>

  "tm.domain"() {
    sym_name = "shards",
    lower = array<i64: 0>,
    upper = array<i64: 2>,
    parameters = [{name = "LAG", value = 1 : i64}]
  } : () -> ()

  %gathered, %gather_done = "tm.allgather_shard"(%source, %t0, %root) {
    id = "ag0",
    reads = ["A[0]"],
    writes = ["gathered_A[0]"],
    root = "gathered_A",
    region = "[0]",
    lookup_key = "gathered_A[0]",
    depends_on = [],
    resource_class = "communication",
    sms = 1 : i64,
    threads = 32 : i64,
    warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)

  %compute_done = "tm.task"(%gathered, %b, %output, %t1, %gather_done) {
    callee = @gemm,
    specialization = "gemm_f32",
    id = "gemm0",
    reads = ["gathered_A[0]", "B"],
    writes = ["C[0]"],
    depends_on = ["ag0"],
    resource_class = "compute",
    sms = 1 : i64,
    threads = 128 : i64,
    warps = 4 : i64,
    shared_memory_bytes = 16384 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event

  "tm.graph"() {
    nodes = [{id = "ag0"}, {id = "gemm0"}],
    time_order = [{from = "ag0", to = "gemm0"}],
    data_deps = [{from = "ag0", kind = "RAW", to = "gemm0"}]
  } : () -> ()

  "tm.plan"() {
    nodes = [{id = "ag0"}, {id = "gemm0"}],
    issue_layers = [{rank = 0 : i64}, {rank = 1 : i64}],
    time_order = [{from = "ag0", to = "gemm0"}],
    data_deps = [{from = "ag0", kind = "RAW", to = "gemm0"}],
    resource_order = [],
    synchronizations = [{from = "ag0", kind = "event", to = "gemm0"}]
  } : () -> ()
}

// CHECK: %[[LAG:.*]] = tm.const 1
// CHECK: %[[T0:.*]] = tm.time.constant 0
// CHECK: %[[T1:.*]] = tm.time.add %[[T0]], %[[LAG]] : !tm.time
// CHECK: %[[ROOT:.*]] = tm.event.none
// CHECK: "tm.binding"
// CHECK: "tm.domain"
// CHECK: "tm.allgather_shard"
// CHECK: "tm.task"
// CHECK: "tm.graph"
// CHECK: "tm.plan"

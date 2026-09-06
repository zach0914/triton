// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph --timely-plan-resources --timely-materialize-synchronization | FileCheck %s

module {
  tt.func private @gemm(%gathered: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                        %output: !tt.ptr<f32>) {
    %lhs = tt.load %gathered : !tt.ptr<f32>
    %rhs = tt.load %b : !tt.ptr<f32>
    %value = arith.addf %lhs, %rhs : f32
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %t2 = tm.time.constant 2
  %root = tm.event.none
  %a0 = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %a1 = "tm.binding"() {
    root = "A", region = "[1]", lookup_key = "A[1]"
  } : () -> !tt.ptr<f32>
  %b = "tm.binding"() {
    root = "B", region = "*", lookup_key = "B"
  } : () -> !tt.ptr<f32>
  %c0 = "tm.binding"() {
    root = "C", region = "[0]", lookup_key = "C[0]"
  } : () -> !tt.ptr<f32>
  %c1 = "tm.binding"() {
    root = "C", region = "[1]", lookup_key = "C[1]"
  } : () -> !tt.ptr<f32>

  %g0, %ag0_done = "tm.allgather_shard"(%a0, %t0, %root) {
    id = "ag0", reads = ["A[0]"], writes = ["G[0]"], depends_on = [],
    root = "G", region = "[0]", lookup_key = "G[0]",
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)
  %g1, %ag1_done = "tm.allgather_shard"(%a1, %t1, %root) {
    id = "ag1", reads = ["A[1]"], writes = ["G[1]"], depends_on = [],
    root = "G", region = "[1]", lookup_key = "G[1]",
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)
  %gemm0_done = "tm.task"(%g0, %b, %c0, %t1, %ag0_done) {
    callee = @gemm, specialization = "gemm_f32", id = "gemm0",
    reads = ["G[0]", "B"], writes = ["C[0]"],
    depends_on = ["ag0"], resource_class = "compute", sms = 1 : i64,
    threads = 128 : i64, warps = 4 : i64, shared_memory_bytes = 16384 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %gemm1_done = "tm.task"(%g1, %b, %c1, %t2, %ag1_done) {
    callee = @gemm, specialization = "gemm_f32", id = "gemm1",
    reads = ["G[1]", "B"], writes = ["C[1]"],
    depends_on = ["ag1"], resource_class = "compute", sms = 1 : i64,
    threads = 128 : i64, warps = 4 : i64, shared_memory_bytes = 16384 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK: id = "ag0"
// CHECK-SAME: tm.issue_rank = 0 : i64
// CHECK: id = "ag1"
// CHECK-SAME: tm.issue_rank = 1 : i64
// CHECK: id = "gemm0"
// CHECK-SAME: tm.issue_rank = 1 : i64
// CHECK: id = "gemm1"
// CHECK-SAME: tm.issue_rank = 2 : i64
// CHECK: "tm.graph"
// CHECK-SAME: data_deps = [{{[^]]*}}from = "ag0"{{[^]]*}}to = "gemm0"{{[^]]*}}from = "ag1"{{[^]]*}}to = "gemm1"
// CHECK-SAME: nodes =
// CHECK-SAME: time_order = [{{[^]]*}}from = "ag0"{{[^]]*}}to = "ag1"{{[^]]*}}from = "ag0"{{[^]]*}}to = "gemm0"
// CHECK: "tm.plan"
// CHECK-SAME: issue_layers = [{{.*}}rank = 0 : i64, tasks = ["ag0"]{{.*}}rank = 1 : i64, tasks = ["ag1", "gemm0"]
// CHECK-SAME: resource_order = []
// CHECK-SAME: synchronizations = [{{[^]]*}}from = "ag0"{{[^]]*}}to = "gemm0"{{[^]]*}}from = "ag1"{{[^]]*}}to = "gemm1"

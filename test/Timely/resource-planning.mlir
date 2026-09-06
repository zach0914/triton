// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph '--timely-plan-resources=max-sms=1' | FileCheck %s

module {
  tt.func private @compute(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %t0 = tm.time.constant 0
  %root = tm.event.none
  %a0 = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %a1 = "tm.binding"() {
    root = "A", region = "[1]", lookup_key = "A[1]"
  } : () -> !tt.ptr<f32>
  %c0 = "tm.binding"() {
    root = "C", region = "[0]", lookup_key = "C[0]"
  } : () -> !tt.ptr<f32>
  %c1 = "tm.binding"() {
    root = "C", region = "[1]", lookup_key = "C[1]"
  } : () -> !tt.ptr<f32>
  %e0 = "tm.task"(%a0, %c0, %t0, %root) {
    callee = @compute, specialization = "compute_f32", id = "compute0",
    reads = ["A[0]"], writes = ["C[0]"],
    depends_on = [], resource_class = "compute", sms = 1 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %e1 = "tm.task"(%a1, %c1, %t0, %root) {
    callee = @compute, specialization = "compute_f32", id = "compute1",
    reads = ["A[1]"], writes = ["C[1]"],
    depends_on = [], resource_class = "compute", sms = 1 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK: "tm.plan"
// CHECK-SAME: issue_layers = [{rank = 0 : i64, tasks = ["compute0", "compute1"]}]
// CHECK-SAME: resource_order = [{from = "compute0", kind = "MVP_SERIAL", to = "compute1"}]
// CHECK-SAME: time_order = []

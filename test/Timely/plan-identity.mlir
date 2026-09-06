// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph --timely-plan-resources --timely-materialize-synchronization | FileCheck %s

module {
  tt.func private @noop(%pointer: !tt.ptr<f32>) {
    tt.return
  }

  %buffer = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %gathered, %gathered_done = "tm.allgather_shard"(%buffer, %t0, %root) {
    root = "G", region = "*", lookup_key = "G", id = "gather",
    reads = ["A"], writes = ["G"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)
  %done0 = "tm.task"(%gathered, %t1, %gathered_done) {
    callee = @noop, specialization = "noop-ptr-f32", id = "invoke0",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %done1 = "tm.task"(%gathered, %t1, %gathered_done) {
    callee = @noop, specialization = "noop-ptr-f32", id = "invoke1",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK: "tm.plan"
// CHECK-SAME: data_deps = [{{.*}}from = "gather", kind = "EVENT", to = "invoke0"{{.*}}from = "gather", kind = "SSA", to = "invoke0"{{.*}}from = "gather", kind = "EVENT", to = "invoke1"{{.*}}from = "gather", kind = "SSA", to = "invoke1"
// CHECK-SAME: nodes = [
// CHECK-SAME: callee = ""
// CHECK-SAME: id = "gather"
// CHECK-SAME: specialization = ""
// CHECK-SAME: callee = "noop"
// CHECK-SAME: id = "invoke0"
// CHECK-SAME: specialization = "noop-ptr-f32"
// CHECK-SAME: callee = "noop"
// CHECK-SAME: id = "invoke1"
// CHECK-SAME: specialization = "noop-ptr-f32"
// CHECK-SAME: resource_order = [{{.*}}from = "invoke0", kind = "MVP_SERIAL", to = "invoke1"
// CHECK-SAME: time_order = [{{.*}}from = "gather", kind = "ISSUE", to = "invoke0"{{.*}}from = "gather", kind = "ISSUE", to = "invoke1"

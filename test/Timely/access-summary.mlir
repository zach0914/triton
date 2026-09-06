// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph | FileCheck %s

module {
  tt.func private @read_write(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }

  %pointer = "tm.binding"() {
    root = "P", region = "*", lookup_key = "P"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %t2 = tm.time.constant 2
  %root = tm.event.none
  %value = arith.constant 1.0 : f32
  %body_done = "tm.task"(%pointer, %t0, %root) {
    callee = @read_write, specialization = "read_write_f32",
    id = "body", reads = ["P"], writes = ["P"],
    depends_on = [], resource_class = "compute", sms = 1 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %x, %e0 = "tm.allgather_shard"(%value, %t0, %root) {
    id = "n0", reads = ["X"], writes = ["A"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
  %y, %e1 = "tm.allgather_shard"(%value, %t1, %root) {
    id = "n1", reads = ["A"], writes = ["X"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
  %z, %e2 = "tm.allgather_shard"(%value, %t2, %root) {
    id = "n2", reads = ["Y"], writes = ["X"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// CHECK: id = "body"
// CHECK: tm.body_read_effects = 1 : i64
// CHECK-SAME: tm.body_write_effects = 1 : i64
// CHECK: "tm.graph"
// CHECK-SAME: from = "n0", kind = "RAW", to = "n1"
// CHECK-SAME: from = "n0", kind = "WAR", to = "n1"
// CHECK-SAME: from = "n0", kind = "WAR", to = "n2"
// CHECK-SAME: from = "n1", kind = "WAW", to = "n2"

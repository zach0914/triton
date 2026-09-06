// RUN: triton-opt --split-input-file %s --timely-normalize-issue-time | FileCheck %s

module {
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %a = arith.constant 1.0 : f32
  %x, %e0 = "tm.allgather_shard"(%a, %t0, %root) {
    id = "near0", reads = ["A[0]"], writes = ["G[0]"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
  %y, %e1 = "tm.allgather_shard"(%a, %t1, %root) {
    id = "near1", reads = ["A[1]"], writes = ["G[1]"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// CHECK: id = "near0"
// CHECK-SAME: tm.issue_rank = 0 : i64
// CHECK: id = "near1"
// CHECK-SAME: tm.issue_rank = 1 : i64

// -----

module {
  %t0 = tm.time.constant 0
  %t100 = tm.time.constant 100
  %root = tm.event.none
  %a = arith.constant 1.0 : f32
  %x, %e0 = "tm.allgather_shard"(%a, %t0, %root) {
    id = "far0", reads = ["A[0]"], writes = ["G[0]"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
  %y, %e1 = "tm.allgather_shard"(%a, %t100, %root) {
    id = "far1", reads = ["A[1]"], writes = ["G[1]"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// CHECK: id = "far0"
// CHECK-SAME: tm.issue_rank = 0 : i64
// CHECK: id = "far1"
// CHECK-SAME: tm.issue_rank = 1 : i64

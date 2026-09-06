// RUN: triton-opt %s --split-input-file --timely-normalize-issue-time --timely-build-dependency-graph --verify-diagnostics | FileCheck %s

// Body-backed accesses to the same typed binding form dependencies without
// repeating reads/writes annotations.
module attributes {tm.test = "overlap"} {
  tt.func private @write(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }

  %buffer = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %written = "tm.task"(%buffer, %t0, %root) {
    callee = @write, specialization = "write-f32", id = "write",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %read = "tm.task"(%buffer, %t1, %root) {
    callee = @read, specialization = "read-f32", id = "read",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK-LABEL: tm.test = "overlap"
// CHECK: id = "write"
// CHECK-SAME: tm.actual_accesses = [{{[^}]*}}argument = 0 : i64{{[^}]*}}lookup_key = "A[0]"{{[^}]*}}region = "[0]"{{[^}]*}}root = "A"{{[^}]*}}writes = true
// CHECK-SAME: tm.alias_roots = 1 : i64
// CHECK-SAME: tm.body_write_effects = 1 : i64
// CHECK-SAME: tm.formal_accesses = [{{.*}}alias_arguments = array<i64: 0>{{.*}}argument = 0 : i64{{.*}}writes = true
// CHECK-SAME: tm.unknown_buffer_regions = 1 : i64
// CHECK: id = "read"
// CHECK-SAME: tm.body_read_effects = 1 : i64
// CHECK: "tm.graph"
// CHECK-SAME: data_deps = [{{.*}}from = "write", kind = "RAW", to = "read"

// -----

// Distinct canonical shard regions of one root are proven disjoint.
module attributes {tm.test = "disjoint"} {
  tt.func private @write(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }

  %buffer0 = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %buffer1 = "tm.binding"() {
    root = "A", region = "[1]", lookup_key = "A[1]"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %done0 = "tm.task"(%buffer0, %t0, %root) {
    callee = @write, specialization = "write-f32", id = "write0",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %done1 = "tm.task"(%buffer1, %t1, %root) {
    callee = @write, specialization = "write-f32", id = "write1",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK-LABEL: tm.test = "disjoint"
// CHECK: "tm.graph"
// CHECK-SAME: data_deps = []

// -----

// A collective result carries destination provenance into its consumer.
module attributes {tm.test = "collective"} {
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }

  %source = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %gathered, %gathered_done = "tm.allgather_shard"(%source, %t0, %root) {
    root = "G", region = "[0]", lookup_key = "G[0]", id = "gather",
    reads = ["A[0]"], writes = ["G[0]"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)
  %done = "tm.task"(%gathered, %t1, %gathered_done) {
    callee = @read, specialization = "read-f32", id = "consume",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK-LABEL: tm.test = "collective"
// CHECK: id = "consume"
// CHECK-SAME: tm.actual_accesses = [{{.*}}lookup_key = "G[0]"{{.*}}region = "[0]"{{.*}}root = "G"
// CHECK: "tm.graph"
// CHECK-SAME: data_deps = [{{.*}}from = "gather", kind = "EVENT", to = "consume"{{.*}}from = "gather", kind = "SSA", to = "consume"

// -----

// Matching root annotations resolve an overlap that opaque region tokens do
// not prove.
module attributes {tm.test = "annotation-supplement"} {
  tt.func private @write(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }

  %lhs = "tm.binding"() {
    root = "A", region = "dynamic-left", lookup_key = "A.left"
  } : () -> !tt.ptr<f32>
  %rhs = "tm.binding"() {
    root = "A", region = "dynamic-right", lookup_key = "A.right"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %written = "tm.task"(%lhs, %t0, %root) {
    callee = @write, specialization = "write-f32", id = "write",
    reads = [], writes = ["A"], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  %read = "tm.task"(%rhs, %t1, %root) {
    callee = @read, specialization = "read-f32", id = "read",
    reads = ["A"], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// CHECK-LABEL: tm.test = "annotation-supplement"
// CHECK: "tm.graph"
// CHECK-SAME: data_deps = [{{.*}}from = "write", kind = "RAW", to = "read"

// -----

module attributes {tm.test = "annotation-contradiction"} {
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }
  %buffer = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op writes annotation 'A' contradicts the body-backed actual operand accesses}}
  %done = "tm.task"(%buffer, %time, %root) {
    callee = @read, specialization = "read-f32", id = "bad-annotation",
    reads = [], writes = ["A"], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module attributes {tm.test = "unknown-region"} {
  tt.func private @write(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }
  %lhs = "tm.binding"() {
    root = "A", region = "dynamic-left", lookup_key = "A.left"
  } : () -> !tt.ptr<f32>
  %rhs = "tm.binding"() {
    root = "A", region = "dynamic-right", lookup_key = "A.right"
  } : () -> !tt.ptr<f32>
  %t0 = tm.time.constant 0
  %t1 = tm.time.constant 1
  %root = tm.event.none
  %written = "tm.task"(%lhs, %t0, %root) {
    callee = @write, specialization = "write-f32", id = "write",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  // expected-error@+1 {{'tm.task' op cannot determine whether body-backed RAW accesses overlap between task 'write' argument #0 (root 'A', region 'dynamic-left') and task 'read' argument #0 (root 'A', region 'dynamic-right'); add matching reads/writes region annotations}}
  %read = "tm.task"(%rhs, %t1, %root) {
    callee = @read, specialization = "read-f32", id = "read",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module attributes {tm.test = "time-inversion"} {
  tt.func private @write(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
  tt.func private @read(%pointer: !tt.ptr<f32>) {
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }
  %buffer = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %late = tm.time.constant 2
  %early = tm.time.constant 1
  %root = tm.event.none
  %written = "tm.task"(%buffer, %late, %root) {
    callee = @write, specialization = "write-f32", id = "late-writer",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
  // expected-error@+1 {{logical issue-time inversion: producer 'late-writer' is issued after consumer 'early-reader'}}
  %read = "tm.task"(%buffer, %early, %root) {
    callee = @read, specialization = "read-f32", id = "early-reader",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

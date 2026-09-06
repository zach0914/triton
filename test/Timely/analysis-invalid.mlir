// RUN: triton-opt --split-input-file %s --timely-normalize-issue-time --timely-build-dependency-graph --verify-diagnostics

module {
  %t0 = tm.time.constant 0
  %root = tm.event.none
  %a = arith.constant 1.0 : f32
  // expected-error@+1 {{'tm.allgather_shard' op depends on missing producer 'missing'}}
  %x, %e0 = "tm.allgather_shard"(%a, %t0, %root) {
    id = "consumer", reads = ["A"], writes = ["G"], depends_on = ["missing"],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// -----

module {
  %runtime = arith.constant 0 : i64
  %time = builtin.unrealized_conversion_cast %runtime : i64 to !tm.time
  %root = tm.event.none
  %a = arith.constant 1.0 : f32
  // expected-error@+1 {{logical issue time must be fully specialized from tm constants}}
  %x, %e0 = "tm.allgather_shard"(%a, %time, %root) {
    id = "dynamic", reads = ["A"], writes = ["G"], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// -----

// expected-error@+1 {{Timely data-dependency graph contains a cycle}}
module {
  %t0 = tm.time.constant 0
  %root = tm.event.none
  %a = arith.constant 1.0 : f32
  %x, %e0 = "tm.allgather_shard"(%a, %t0, %root) {
    id = "left", reads = ["A[0]"], writes = ["G[0]"], depends_on = ["right"],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
  %y, %e1 = "tm.allgather_shard"(%a, %t0, %root) {
    id = "right", reads = ["A[1]"], writes = ["G[1]"], depends_on = ["left"],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// -----

module {
  tt.func private @compute(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %early = tm.time.constant 0
  %late = tm.time.constant 1
  %root = tm.event.none
  %a = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  %c = "tm.binding"() {
    root = "C", region = "[0]", lookup_key = "C[0]"
  } : () -> !tt.ptr<f32>
  %x, %e0 = "tm.allgather_shard"(%a, %late, %root) {
    id = "producer", reads = ["A[0]"], writes = ["G[0]"], depends_on = [],
    root = "G", region = "[0]", lookup_key = "G[0]",
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (!tt.ptr<f32>, !tm.event)
  // expected-error@+1 {{logical issue-time inversion: producer 'producer' is issued after consumer 'consumer'}}
  %e1 = "tm.task"(%x, %c, %early, %e0) {
    callee = @compute, specialization = "compute_f32", id = "consumer",
    reads = ["G[0]"], writes = ["C[0]"],
    depends_on = ["producer"], resource_class = "compute", sms = 1 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

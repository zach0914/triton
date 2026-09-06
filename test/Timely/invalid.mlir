// RUN: triton-opt --split-input-file %s --verify-diagnostics

module {
  // expected-error@+1 {{'tm.domain' op requires every lower bound to be smaller than upper}}
  "tm.domain"() {
    sym_name = "empty",
    lower = array<i64: 2>, upper = array<i64: 2>, parameters = []
  } : () -> ()
}

// -----

module {
  // expected-error@+1 {{'tm.domain' op requires unique scheduling parameter names}}
  "tm.domain"() {
    sym_name = "duplicate_params",
    lower = array<i64: 0>, upper = array<i64: 2>,
    parameters = [
      {name = "LAG", value = 1 : i64},
      {name = "LAG", value = 2 : i64}
    ]
  } : () -> ()
}

// -----

module {
  %time = tm.time.constant 0
  %root = tm.event.none
  %source = arith.constant 1.0 : f32
  // expected-error@+1 {{'tm.allgather_shard' op requires a destination in writes}}
  %data, %done = "tm.allgather_shard"(%source, %time, %root) {
    id = "ag0", reads = ["A[0]"], writes = [], depends_on = [],
    resource_class = "communication", sms = 1 : i64, threads = 32 : i64,
    warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (f32, !tm.time, !tm.event) -> (f32, !tm.event)
}

// -----

module {
  tt.func private @gemm(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %time = tm.time.constant 0
  %root = tm.event.none
  %source = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %output = "tm.binding"() {
    root = "C", region = "*", lookup_key = "C"
  } : () -> !tt.ptr<f32>
  // expected-error@+1 {{'tm.task' op cannot depend on itself}}
  %done = "tm.task"(%source, %output, %time, %root) {
    callee = @gemm, specialization = "gemm_f32", id = "gemm0",
    reads = ["A"], writes = ["C"],
    depends_on = ["gemm0"], resource_class = "compute", sms = 1 : i64,
    threads = 128 : i64, warps = 4 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @gemm(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %time = tm.time.constant 0
  %root = tm.event.none
  %source = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %output = "tm.binding"() {
    root = "C", region = "*", lookup_key = "C"
  } : () -> !tt.ptr<f32>
  // expected-error@+1 {{'tm.task' op unknown accesses are unsupported; add reads/writes annotations}}
  %done = "tm.task"(%source, %output, %time, %root) {
    callee = @gemm, specialization = "gemm_f32", id = "gemm0",
    reads = [], writes = [], depends_on = [],
    resource_class = "compute", sms = 1 : i64, threads = 128 : i64,
    warps = 4 : i64, shared_memory_bytes = 0 : i64,
    unknown_access = true
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

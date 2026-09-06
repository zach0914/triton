// RUN: triton-opt --split-input-file %s --timely-normalize-issue-time --timely-build-dependency-graph --timely-plan-resources --timely-materialize-synchronization --verify-diagnostics

module {
  %input = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'dangling' callee '@missing' does not resolve to a symbol}}
  %done = "tm.task"(%input, %time, %root) {
    callee = @missing, specialization = "missing_f32", id = "dangling",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @kernel(%input: !tt.ptr<f32>) {
    tt.return
  }
  %input = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op requires one result}}
  %fake, %done = "tm.task"(%input, %time, %root) {
    callee = @kernel, specialization = "kernel_f32", id = "fake-result",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> (f32, !tm.event)
}

// -----

module {
  tt.func private @kernel(%input: !tt.ptr<f32>) {
    tt.return
  }
  %zero = arith.constant 0 : i64
  %input = tt.int_to_ptr %zero : i64 -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'unbound' callee '@kernel' pointer argument 0 must originate from a tm.binding}}
  %done = "tm.task"(%input, %time, %root) {
    callee = @kernel, specialization = "kernel_f32", id = "unbound",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  "tm.domain"() {
    sym_name = "not_a_function", lower = array<i64: 0>,
    upper = array<i64: 1>, parameters = []
  } : () -> ()
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'wrong-kind' callee '@not_a_function' must resolve to a tt.func, but resolved to 'tm.domain'}}
  %done = "tm.task"(%time, %root) {
    callee = @not_a_function, specialization = "wrong_kind", id = "wrong-kind",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @external(!tt.ptr<f32>)
  %input = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'external' callee '@external' must have a body}}
  %done = "tm.task"(%input, %time, %root) {
    callee = @external, specialization = "external_f32", id = "external",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @two_args(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>) {
    tt.return
  }
  %input = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'arity' callee '@two_args' expects 2 dynamic arguments, but task provides 1}}
  %done = "tm.task"(%input, %time, %root) {
    callee = @two_args, specialization = "two_args_f32", id = "arity",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @expects_f16(%input: !tt.ptr<f16>) {
    tt.return
  }
  %input = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'type' callee '@expects_f16' argument 0 type mismatch: expected '!tt.ptr<f16>', but got '!tt.ptr<f32>'}}
  %done = "tm.task"(%input, %time, %root) {
    callee = @expects_f16, specialization = "expects_f16", id = "type",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @returns_value() -> f32 {
    %value = arith.constant 0.0 : f32
    tt.return %value : f32
  }
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op task 'non-void' callee '@returns_value' must return void, but returns 'f32'}}
  %done = "tm.task"(%time, %root) {
    callee = @returns_value, specialization = "returns_f32", id = "non-void",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tm.time, !tm.event) -> !tm.event
}

// -----

module {
  tt.func private @kernel() {
    tt.return
  }
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op requires a non-empty specialization identity}}
  %done = "tm.task"(%time, %root) {
    callee = @kernel, specialization = "", id = "empty-specialization",
    reads = [], writes = [], depends_on = [], resource_class = "compute",
    sms = 1 : i64, threads = 32 : i64, warps = 1 : i64,
    shared_memory_bytes = 0 : i64
  } : (!tm.time, !tm.event) -> !tm.event
}

// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph --verify-diagnostics

module {
  tt.func private @writes(%pointer: !tt.ptr<f32>) {
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }

  %pointer = "tm.binding"() {
    root = "P", region = "*", lookup_key = "P"
  } : () -> !tt.ptr<f32>
  %time = tm.time.constant 0
  %root = tm.event.none
  // expected-error@+1 {{'tm.task' op writes annotation 'Q' contradicts the body-backed actual operand accesses}}
  %done = "tm.task"(%pointer, %time, %root) {
    callee = @writes, specialization = "writes_f32",
    id = "wrong_write", reads = [], writes = ["Q"],
    depends_on = [], resource_class = "compute", sms = 1 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

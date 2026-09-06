// RUN: triton-opt %s --timely-normalize-issue-time --timely-build-dependency-graph '--timely-plan-resources=max-sms=1' --verify-diagnostics

module {
  tt.func private @compute(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }

  %t0 = tm.time.constant 0
  %root = tm.event.none
  %a = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
  %c = "tm.binding"() {
    root = "C", region = "*", lookup_key = "C"
  } : () -> !tt.ptr<f32>
  // expected-error@+1 {{'tm.task' op resource request exceeds target capacity}}
  %e0 = "tm.task"(%a, %c, %t0, %root) {
    callee = @compute, specialization = "compute_f32", id = "too_large",
    reads = ["A"], writes = ["C"],
    depends_on = [], resource_class = "compute", sms = 2 : i64,
    threads = 32 : i64, warps = 1 : i64, shared_memory_bytes = 0 : i64
  } : (!tt.ptr<f32>, !tt.ptr<f32>, !tm.time, !tm.event) -> !tm.event
}

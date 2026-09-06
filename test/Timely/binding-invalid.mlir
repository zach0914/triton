// RUN: triton-opt --split-input-file %s --verify-diagnostics

module {
  // expected-error@+1 {{'tm.binding' op requires a non-empty root}}
  %value = "tm.binding"() {
    root = "", region = "*", lookup_key = "A"
  } : () -> !tt.ptr<f32>
}

// -----

module {
  // expected-error@+1 {{'tm.binding' op requires a non-empty region}}
  %value = "tm.binding"() {
    root = "A", region = "", lookup_key = "A"
  } : () -> !tt.ptr<f32>
}

// -----

module {
  // expected-error@+1 {{'tm.binding' op requires a non-empty lookup_key}}
  %value = "tm.binding"() {
    root = "A", region = "*", lookup_key = ""
  } : () -> !tt.ptr<f32>
}

// -----

module {
  // expected-error@+1 {{'tm.binding' op cannot bind a Timely scheduling type}}
  %value = "tm.binding"() {
    root = "A", region = "*", lookup_key = "A"
  } : () -> !tm.event
}

// -----

module {
  %first = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
  // expected-error@+1 {{'tm.binding' op requires a unique lookup_key; 'A[0]' was already declared}}
  %duplicate = "tm.binding"() {
    root = "A", region = "[1]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f32>
}

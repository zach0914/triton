// RUN: triton-opt %s | FileCheck %s

module {
  %buffer = "tm.binding"() {
    root = "A", region = "[0]", lookup_key = "A[0]"
  } : () -> !tt.ptr<f16>
  %length = "tm.binding"() {
    root = "N", region = "*", lookup_key = "N"
  } : () -> i32
}

// CHECK: %[[BUFFER:.*]] = "tm.binding"() <{lookup_key = "A[0]", region = "[0]", root = "A"}> : () -> !tt.ptr<f16>
// CHECK: %[[LENGTH:.*]] = "tm.binding"() <{lookup_key = "N", region = "*", root = "N"}> : () -> i32

#ifndef TIMELY_DIALECT
#define TIMELY_DIALECT

include "mlir/IR/OpBase.td"

def Timely_Dialect : Dialect {
  let name = "tm";
  let summary = "Logical issue-time scheduling for Triton tasks";
  let description = [{
    Timely represents issue-time partial order, data dependencies, and resource
    requirements as distinct properties of a static task graph.
  }];
  let cppNamespace = "::mlir::triton::timely";
  let useDefaultTypePrinterParser = 1;
}

class Timely_Op<string mnemonic, list<Trait> traits = []>
    : Op<Timely_Dialect, mnemonic, traits>;

#endif // TIMELY_DIALECT

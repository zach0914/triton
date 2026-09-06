#ifndef TIMELY_TYPES
#define TIMELY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Timely/IR/TimelyDialect.td"

class Timely_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Timely_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Timely_TimeType : Timely_Type<"Time", "time"> {
  let summary = "A logical issue-time key";
}

def Timely_ConstType : Timely_Type<"Const", "const"> {
  let summary = "A compile-time scheduling constant";
}

def Timely_EventType : Timely_Type<"Event", "event"> {
  let summary = "Completion of an asynchronous task";
}

#endif // TIMELY_TYPES

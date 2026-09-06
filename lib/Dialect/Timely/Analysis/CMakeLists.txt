add_triton_library(TimelyAnalysis
  TaskAccess.cpp

  LINK_LIBS PUBLIC
    TimelyIR
    TritonAnalysis
    TritonIR
    TritonGPUIR
)

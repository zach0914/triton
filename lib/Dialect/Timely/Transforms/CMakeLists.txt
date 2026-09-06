add_triton_library(TimelyTransforms
  Passes.cpp

  DEPENDS
    TimelyTransformsIncGen

  LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransforms
    TimelyAnalysis
    TimelyIR
)

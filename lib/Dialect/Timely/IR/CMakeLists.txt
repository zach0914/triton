add_triton_library(TimelyIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
    TimelyTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    TritonIR
)

# Pitch detector.
add_library(pitch STATIC
    src/pitch.c
)

target_include_directories(pitch
    PUBLIC  include
    PRIVATE src
)

target_compile_options(pitch PRIVATE ${TUNER_LIB_WARNINGS})

target_link_libraries(pitch PRIVATE m)

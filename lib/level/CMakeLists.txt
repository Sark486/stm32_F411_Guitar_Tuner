# Level meter: DC blocker + RMS over a time window, in dBFS.
add_library(level STATIC
    src/level.c
)

# PUBLIC: level's own sources and every target that links level see include/.
target_include_directories(level PUBLIC include)

target_compile_options(level PRIVATE ${TUNER_LIB_WARNINGS})

target_link_libraries(level PRIVATE m)

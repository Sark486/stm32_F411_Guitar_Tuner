# Portable C modules, shared by two builds:
#   - the firmware (root CMakeLists.txt, arm-none-eabi-gcc), and
#   - the host tests (tests/CMakeLists.txt, the PC's own compiler).

set(TUNER_LIB_WARNINGS
    -Wall
    -Wextra
    -Wdouble-promotion   # float silently promoted to double (slow on the M4F)
    -Wfloat-conversion   # double -> float, float -> int without a cast
    -Wconversion         # any implicit conversion that can change a value
    -Wshadow             # a local hiding another variable
)

add_subdirectory(level)
add_subdirectory(pitch)

cmake_minimum_required(VERSION 3.22)

# Host (PC) test: the C pitch module checked against the Python golden vectors.
# A separate CMake project from the firmware: one build directory has exactly one
# compiler, and this one uses the PC's own compiler, not arm-none-eabi-gcc.
# From this folder:
#
#   cmake --preset host            configure (build/host)
#   cmake --build --preset host    build
#   ctest --preset host            run

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # build/host/compile_commands.json for clangd

project(tuner_host_tests LANGUAGES C)

enable_testing()   # must come before add_test()

# The portable modules, exactly as the firmware uses them. lib/ is outside this
# folder, so CMake needs a name for its build folder: the second argument.
add_subdirectory(../lib lib)

# The golden-vector test: a plain C program, no test framework. It also includes
# the module's internal header, so it gets lib/pitch/src on its include path.
add_executable(test_pitch test_pitch.c)
target_link_libraries(test_pitch PRIVATE pitch)
target_include_directories(test_pitch PRIVATE ../lib/pitch/src)
target_compile_options(test_pitch PRIVATE -Wall -Wextra -Wshadow)
# Absolute path to the vectors: the test runs from the build folder, so a
# relative path would break.
target_compile_definitions(test_pitch PRIVATE VECTORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vectors")

# CTest judges the test only by its exit code: main() returns 0 when every
# check passed, non-zero otherwise.
add_test(NAME test_pitch COMMAND test_pitch)

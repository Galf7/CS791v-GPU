project(PA0)
cmake_minimum_required(VERSION 2.8)

# using cuda
find_package(CUDA QUIET REQUIRED)

# set directories
set(INCLUDES ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDES})
file(GLOB_RECURSE SOURCES "src/*.cu")

cuda_add_executable(
        ${PROJECT_NAME}
        ${SOURCES}
        ${INCLUDES}
)
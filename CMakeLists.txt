cmake_minimum_required(VERSION 3.14)
project(MPCProject)

set(CMAKE_CXX_STANDARD 17)

# CasADi include and lib
include_directories(
    ${PROJECT_SOURCE_DIR}/external/casadi-main/install/casadi/include
    ${PROJECT_SOURCE_DIR}/cpp/include
)
link_directories(${PROJECT_SOURCE_DIR}/external/casadi-main/install/casadi)

# Add executable and your source files
add_executable(main
    cpp/main.cpp
)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
# Link with CasADi
target_link_libraries(main casadi)

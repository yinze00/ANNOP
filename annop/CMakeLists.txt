find_package(TensorFlow REQUIRED)

add_library(
    annops 
    SHARED
    kernel/ZeroOutOp.cc
    ops/zero_out.cc
)
target_compile_options(annops PRIVATE -fPIC)

# target_link_libraries(annops )
target_include_directories(annops PRIVATE ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(annops PRIVATE ${TensorFlow_LIBRARIES})

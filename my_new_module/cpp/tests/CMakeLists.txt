cmake_minimum_required(VERSION 3.14)
project(my_new_module_tests)

# Find dependencies
find_package(GTest REQUIRED)
find_package(my_new_module CONFIG REQUIRED)

# Add test executable
add_executable(test_my_new_module
    test_my_new_module.cpp
)

# Link libraries
target_link_libraries(test_my_new_module
    PRIVATE
        my_new_module::my_new_module
        GTest::GTest
        GTest::Main
)

# Add test
add_test(NAME my_new_module_test
    COMMAND test_my_new_module
)

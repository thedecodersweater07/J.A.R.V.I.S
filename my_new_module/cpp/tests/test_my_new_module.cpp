#include <gtest/gtest.h>
#include <jarvis/my_new_module.hpp>

TEST(MyNewModuleTest, BasicTest) {
    jarvis::MyNewModule module;
    auto result = module.process("test");
    EXPECT_NE(result.find("Processed by my_new_module:"), std::string::npos);
}

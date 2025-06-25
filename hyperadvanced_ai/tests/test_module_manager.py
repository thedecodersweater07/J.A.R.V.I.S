import unittest
from hyperadvanced_ai.core.module_manager import ModuleManager

class TestModuleManager(unittest.TestCase):
    def test_load_module(self):
        manager = ModuleManager()
        mod = manager.load_module('hyperadvanced_ai.modules.example_module', 'ExampleModule')
        self.assertIsNotNone(mod)

if __name__ == '__main__':
    unittest.main()

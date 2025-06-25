import unittest
from hyperadvanced_ai.utils.dynamic_loader import dynamic_import

class TestDynamicLoader(unittest.TestCase):
    def test_dynamic_import(self):
        mod = dynamic_import('hyperadvanced_ai.modules.example_module')
        self.assertIsNotNone(mod)

if __name__ == '__main__':
    unittest.main()

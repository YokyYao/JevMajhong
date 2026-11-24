"""Expose the frozen three-function checks to unittest discovery."""
from pathlib import Path
import runpy
import unittest


def load_tests(loader, existing, pattern):
    namespace = runpy.run_path(Path(__file__).with_name('test_curriculum_kan_source_teacher_v1.py'))
    names = sorted(name for name, value in namespace.items()
                   if name.startswith('test_') and callable(value))
    if len(names) != 3:
        raise ValueError('The frozen source/teacher contract contains exactly three checks')
    return unittest.TestSuite(unittest.FunctionTestCase(namespace[name], description=name) for name in names)

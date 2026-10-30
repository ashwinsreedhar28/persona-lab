"""Compatibility entrypoint; implementation lives in tools/build_refusal.py."""
import importlib
import sys
implementation=importlib.import_module('tools.build_refusal')
if __name__=='__main__':
    implementation.main()
else:
    sys.modules[__name__]=implementation

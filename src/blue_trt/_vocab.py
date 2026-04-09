import os as _os, sys as _sys
_src = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _src not in _sys.path:
    _sys.path.insert(0, _src)
from _blue_vocab import *  # noqa: F401, F403
del _os, _sys, _src

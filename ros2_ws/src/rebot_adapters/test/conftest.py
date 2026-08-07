"""Make the package importable without installation or ROS.

The cores under rebot_adapters/core are rclpy-free by design; these tests
run on a plain Python interpreter (no ROS environment sourced).
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

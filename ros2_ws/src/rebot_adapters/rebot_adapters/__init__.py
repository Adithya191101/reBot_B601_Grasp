"""reBot B601-DM canonical ROS 2 adapters.

Keep this module import-light: importing :mod:`rebot_adapters` (or
``rebot_adapters.core``) must not pull in rclpy. The node modules
(``*_adapter_node``) are imported only by their console-script entry
points inside a ROS environment.
"""

__all__ = ["core"]

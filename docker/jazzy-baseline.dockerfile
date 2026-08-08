# M1/M3' baseline container: plain ROS 2 Jazzy + MoveIt for the Seeed mock
# demo and kinematics tests. NOT the Isaac ROS dev container (that comes from
# the isaac-ros CLI and is required from M5/M6 on); this one needs no host
# sudo to build or run.
FROM osrf/ros:jazzy-desktop

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-moveit \
    ros-jazzy-moveit-resources \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gripper-controllers \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-pinocchio \
    ros-jazzy-xacro \
    ros-jazzy-rosbag2-storage-mcap \
    ros-jazzy-tf-transformations \
    python3-pytest \
    liburdfdom-tools \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

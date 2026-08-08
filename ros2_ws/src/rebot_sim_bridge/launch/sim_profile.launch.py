"""M5 sim-profile launch: adapters (sim mode) + sim-JTC shims + planner.

Runs inside the rebot-jazzy-baseline container (repo mounted at /work,
--network host, ROS_DOMAIN_ID=42, default FastDDS) against the host-side
Isaac bridge (scripts/b601_sim_bridge.py).  Every node runs with
use_sim_time:=true -- Isaac Sim is the single /clock owner (sim_topics.yaml
rule).

Topology (config/sim_topics.yaml):

    planner  /rebot_planner/move_to_pose (rebot_planner_msgs/MoveToPose)
       |
    /rebot_controller/follow_joint_trajectory      (canonical)
       |
    trajectory adapter (mode=sim)
       |
    /rebot_sim_arm_controller/follow_joint_trajectory   (frozen sim name)
       |
    sim-JTC shim (arm) ----\
                            +--> /isaac_joint_commands --> Isaac articulation
    sim-JTC shim (jaw) ----/
       |
    /gripper_controller/follow_joint_trajectory <- gripper adapter (mode=sim)
                                                    <- /rebot_controller/gripper_command

    /isaac_joint_states -> joint-state adapter (mode=sim) -> /joint_states

The adapters' sim_input/sim_action parameters are overridden here rather
than edited in adapters.yaml: the rearm environment's Isaac-side names are a
recorded decision (sim_topics.yaml), the package defaults stay the design-doc
values.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ISAAC_JOINT_STATES = "/isaac_joint_states"
ISAAC_JOINT_COMMANDS = "/isaac_joint_commands"
ARM_SIM_ACTION = "/rebot_sim_arm_controller/follow_joint_trajectory"
GRIPPER_SIM_ACTION = "/gripper_controller/follow_joint_trajectory"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
JAW_JOINTS = ["gripper_joint1", "gripper_joint2"]


def generate_launch_description() -> LaunchDescription:
    urdf_arg = DeclareLaunchArgument(
        "urdf_path", default_value="/work/urdf/rebot_b601dm_canonical.urdf",
        description="canonical URDF (repo mounted at /work in the container)")
    urdf_path = LaunchConfiguration("urdf_path")
    sim_time = {"use_sim_time": True}

    return LaunchDescription([
        urdf_arg,

        Node(
            package="rebot_adapters",
            executable="rebot_joint_state_adapter",
            name="rebot_joint_state_adapter",
            output="screen",
            parameters=[sim_time, {
                "mode": "sim",
                "sim_input_topic": ISAAC_JOINT_STATES,
                # 60 Hz sim publisher; 0.20 s stale gate keeps design margin.
                "publish_rate_hz": 60.0,
            }],
        ),

        Node(
            package="rebot_adapters",
            executable="rebot_trajectory_adapter",
            name="rebot_trajectory_adapter",
            output="screen",
            parameters=[sim_time, {
                "mode": "sim",
                "sim_action": ARM_SIM_ACTION,
            }],
        ),

        Node(
            package="rebot_adapters",
            executable="rebot_gripper_adapter",
            name="rebot_gripper_adapter",
            output="screen",
            parameters=[sim_time, {
                "mode": "sim",
                "sim_action": GRIPPER_SIM_ACTION,
            }],
        ),

        # FJT -> /isaac_joint_commands shims (see sim_jtc_shim_node.py).
        Node(
            package="rebot_sim_bridge",
            executable="sim_jtc_shim",
            name="sim_arm_jtc_shim",
            output="screen",
            parameters=[sim_time, {
                "action_name": ARM_SIM_ACTION,
                "joint_names": ARM_JOINTS,
                "command_topic": ISAAC_JOINT_COMMANDS,
                "state_topic": ISAAC_JOINT_STATES,
                "command_rate_hz": 60.0,
                "goal_tolerance": 0.02,        # rad, M5 parity gate
                "settle_timeout_sec": 3.0,
            }],
        ),
        Node(
            package="rebot_sim_bridge",
            executable="sim_jtc_shim",
            name="sim_gripper_jtc_shim",
            output="screen",
            parameters=[sim_time, {
                "action_name": GRIPPER_SIM_ACTION,
                "joint_names": JAW_JOINTS,
                "command_topic": ISAAC_JOINT_COMMANDS,
                "state_topic": ISAAC_JOINT_STATES,
                "command_rate_hz": 60.0,
                "goal_tolerance": 0.005,       # metres per jaw
                "settle_timeout_sec": 3.0,
            }],
        ),

        Node(
            package="rebot_planner",
            executable="rebot_planner_node",
            name="rebot_planner",
            output="screen",
            parameters=[sim_time, {
                "urdf_path": urdf_path,
                # remaining planner params: package defaults (planner.yaml
                # values are the defaults declared by the node).
            }],
        ),
    ])

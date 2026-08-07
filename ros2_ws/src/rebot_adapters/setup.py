from setuptools import find_packages, setup

package_name = "rebot_adapters"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", [
            "config/moveit_controllers.yaml",
            "config/adapters.yaml",
            "config/gripper_calibration.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adithya",
    maintainer_email="nirty.4u@gmail.com",
    description=(
        "Canonical reBot B601-DM adapters (joint state, trajectory, gripper) "
        "with rclpy-free validation cores."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rebot_joint_state_adapter = "
            "rebot_adapters.joint_state_adapter_node:main",
            "rebot_trajectory_adapter = "
            "rebot_adapters.trajectory_adapter_node:main",
            "rebot_gripper_adapter = "
            "rebot_adapters.gripper_adapter_node:main",
        ],
    },
)

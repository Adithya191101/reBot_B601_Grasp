from setuptools import find_packages, setup

package_name = "rebot_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", [
            "config/cell_geometry.yaml",
            "config/planner.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adithya",
    maintainer_email="nirty.4u@gmail.com",
    description=(
        "Pinocchio-based IK/planning for the reBot B601-DM (MoveIt "
        "replacement): rclpy-free planning cores + thin MoveToPose node."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rebot_planner_node = rebot_planner.planner_node:main",
        ],
    },
)

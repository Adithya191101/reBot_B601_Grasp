from setuptools import find_packages, setup

package_name = "rebot_sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/sim_profile.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Adithya",
    maintainer_email="nirty.4u@gmail.com",
    description=(
        "M5 sim-profile glue: FJT->JointState shim controllers for the Isaac "
        "bridge closed loop, and the sim-profile launch file."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_jtc_shim = rebot_sim_bridge.sim_jtc_shim_node:main",
        ],
    },
)

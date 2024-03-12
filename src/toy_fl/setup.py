from setuptools import find_packages, setup

package_name = "toy_fl"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="markhaoxiang",
    maintainer_email="mark.haoxiang@gmail.com",
    description="Runs a toy federated learning dataset as a simulated robotic training task",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "downloader = toy_fl.dataset_downloader:main",
            "publisher = toy_fl.dataset_publisher:main",
            "client = toy_fl.client:main",
            "gym_controller = toy_fl.gym_controller:main",
            "transition_publisher_placeholder = toy_fl.placeholder.transition_publisher_placeholder:main",
            "policy_placeholder = toy_fl.placeholder.policy_placeholder:main",
            "dqn_actor = toy_fl.gym_sim.dqn.dqn_actor:main",
            "dqn_client = toy_fl.gym_sim.dqn.dqn_client:main",
        ],
    },
)

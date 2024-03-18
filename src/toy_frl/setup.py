from setuptools import find_packages, setup

package_name = 'toy_frl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='markhaoxiang',
    maintainer_email='mark.haoxiang@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "transition_publisher_placeholder = toy_frl.placeholder.transition_publisher_placeholder:main",
            "policy_placeholder = toy_frl.placeholder.policy_placeholder:main",
            "gym_controller = toy_frl.gym_controller:main",  
            "dqn_actor = toy_frl.dqn.dqn_actor:main",
            "dqn_client = toy_frl.dqn.dqn_client:main",
        ],
    },
)

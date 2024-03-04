from setuptools import find_packages, setup

package_name = 'toy_fl_publisher'

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
    description='Publishes toy federated learning datasets as a simulated sensor',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'downloader = toy_fl_publisher.dataset_downloader:main',
            'publisher = toy_fl_publisher.dataset_publisher:main',
        ],
    },
)

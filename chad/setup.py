from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'chad'

otherfiles = [
    (os.path.join('share', package_name, 'launch'),
        glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'rviz'),
        glob('rviz/*')),
    (os.path.join('share', package_name, 'meshes'),
        glob('meshes/*')),
    (os.path.join('share', package_name, 'urdf'),
        glob('urdf/*')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
    ] + otherfiles,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@Bender.local',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'touchtable_bottles = chad.touchtable_bottles:main',
            'trajectory_node = chad.trajectory_node:main',
            'brain_node = chad.brain_node:main',
            'go_to_zero = chad.go_to_zero:main',
            'sixdof_touchtable = chad.sixdof_touchtable:main',
            'go_to_zero_spline = chad.go_to_zero_spline:main',
            'sixdof = chad.sixdof:main',
            'gravity = chad.gravity:main',
            'plothebi = chad.plothebi:main',
        ],
    },
)

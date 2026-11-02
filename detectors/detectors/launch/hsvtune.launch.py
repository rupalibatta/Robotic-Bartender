"""Launch the USB camera node and HSV tuning utility.

This launch file is intended show how the pieces come together.
Please copy the relevant pieces.

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure the USB camera node
    node_usbcam = Node(
        name       = 'usb_cam', 
        package    = 'usb_cam',
        executable = 'usb_cam_node_exe',
        namespace  = 'usb_cam',
        output     = 'screen',
        parameters = [{'camera_name':  'logitech'},
                      {'video_device': '/dev/video0'},
                      {'pixel_format': 'yuyv2rgb'},
                      {'image_width':  640},
                      {'image_height': 480},
                      {'framerate':    15.0},
                      {'auto_white_balance': False},
                      {'white_balance': 4000},
                      {'autoexposure': False},
                      {'exposure': 255},
                      {'gain': 0}
                      ])

    # Configure the HSV tuning utility node
    node_hsvtune = Node(
        name       = 'hsvtuner', 
        package    = 'detectors',
        executable = 'hsvtune',
        output     = 'screen',
        remappings = [('/image_raw', '/usb_cam/image_raw')])

    # Configure the camera settings
    node_cameracontrol = Node(
        name       = 'cameracontrol', 
        package    = 'detectors',
        executable = 'cameracontrol',
        output     = 'screen')


    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the nodes.
        node_usbcam,
        node_hsvtune,
        node_cameracontrol,
    ])

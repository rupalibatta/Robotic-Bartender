#!/usr/bin/env python3
#
#   balldetector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Node:           /balldetector
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /balldetector/binary        Intermediate binary image
#                   /balldetector/image_raw     Debug (marked up) image
#
import cv2
import numpy as np
import math

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point
from geometry_msgs.msg import Pose
from detectors.mapping import DemoNode

#
#  Detector Node Class
#
class DetectorNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.mapper = DemoNode("mapper")
        self.x0 = 0.139
        self.y0 = 0.648

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.hsvlimits = np.array([[0, 30], [91, 210], [190, 255]])

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)
        self.pub_point = self.create_publisher(Point, name+'/disk', 3)
        self.pub_pose = self.create_publisher(Pose, name+"/strip", 3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")

    def publish_disc(self, image, u, v):
        xy = self.mapper.pixelToWorld(image, u, v, self.x0, self.y0)
        if xy is not None:
            msg = Point()
            msg.x = float(xy[0])
            msg.y = float(xy[1])
            msg.z = 0.0
            self.pub_point.publish(msg)

    def publish_strip(self, image, u, v, theta):
        xy = self.mapper.pixelToWorld(image, u, v, self.x0, self.y0)
        if xy is not None:
            msg = Pose()
            msg.position.x = float(xy[0])
            msg.position.y = float(xy[1])
            msg.position.z = 0.0
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = np.sin(theta/2)
            msg.orientation.w = np.cos(theta/2)
            self.pub_pose.publish(msg)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    def angle_minarearect(self, contour):
        rect = cv2.minAreaRect(contour)
        ((um, vm), (wm, hm), angle) = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        # figure out which is the longest edge
        longest_vec = None
        longest_len2 = -1.0
        for i in range(4):
            p0 = box[i]
            p1 = box[(i+1) % 4]
            v = p1-p0
            L2 = v[0]**2 + v[1]**2
            if L2 > longest_len2:
                longest_len2 = L2
                longest_vec = v
        theta = math.atan2(longest_vec[1], longest_vec[0])
        L = math.sqrt(longest_len2)
        t_hat = longest_vec/L
        n_hat = np.array([-t_hat[1], t_hat[0]])

        return (um, vm), theta, n_hat, box.astype(np.int32)

    
    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Help to determine the HSV range...
        if True:
            # Draw the center lines.  Note the row is the first dimension.
            frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            self.get_logger().info(
                "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        
        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary = cv2.inRange(hsv, self.hsvlimits[:,0], self.hsvlimits[:,1])

        # Erode and Dilate. Definitely adjust the iterations!
        iter = 2
        binary = cv2.erode( binary, None, iterations=iter)
        binary = cv2.dilate(binary, None, iterations=2*iter)
        binary = cv2.erode( binary, None, iterations=iter)


        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours on the original image for debugging.
        cv2.drawContours(frame, contours, -1, self.blue, 2)

        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours...
        if len(contours) > 0:
            min_area = 150
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area < min_area:
                    continue

                # Find the enclosing circle (convert to pixel values)
                ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
                ur     = int(ur)
                vr     = int(vr)
                radius = int(radius)

                circle_area = np.pi * radius * radius
                ballness = circle_area / contour_area

                ballness_threshold = 1.5
                # Draw the circle (yellow) and centroid (red) on the
                    # original image.

                
                if ballness > ballness_threshold:
                    # then this is a strip
                    (center_uv, theta, n_hat, box) = self.angle_minarearect(contour)
                    (um, vm) = center_uv
                    cv2.drawContours(frame, [box], 0, self.yellow, 2)
                    L=60
                    cx,cy = int(um), int(vm)
                    cv2.line(frame, (cx, cy), (int(cx+L*np.cos(theta)), int(cy+L*np.sin(theta))), self.green, 2)
                    cv2.line(frame, (cx, cy), (int(cx-L*np.sin(theta)), int(cy+L*np.cos(theta))), self.red, 2)
                    self.get_logger().info(
                         f"Found long strip with axis angle of theta = {theta:.3f} rad")
                    self.publish_strip(frame, um, vm, theta)
                  
                else:
                    # then this is a ball
                    # Report. - ball (disk) detected
                    cv2.circle(frame, (ur, vr), int(radius), self.yellow,  2)
                    cv2.circle(frame, (ur, vr), 1,           self.red,    -1)
                 
                    self.get_logger().info(
                        "Found ball enclosed by radius %d about (%d,%d)" %
                        (radius, ur, vr))
                    self.publish_disc(frame, ur, vr)
                
        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('balldetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

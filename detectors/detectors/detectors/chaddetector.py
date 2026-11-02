#!/usr/bin/env python3
#
#   chaddetector.py
#
#   Detect purple and blue bottles for the Chad cocktail robot.
#   Publishes world-frame positions as ObjectArray (and debug images).
#
#   Node:           /chaddetector
#   Subscribers:    /image_raw
#   Publishers:     /chaddetector/objects       project_msgs/ObjectArray
#                   /chaddetector/image_raw     Debug image
#                   /chaddetector/binary        Combined binary
#                   /chaddetector/purple_bottle geometry_msgs/Point (legacy)
#                   /chaddetector/blue_bottle   geometry_msgs/Point (legacy)
#
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point
from geometry_msgs.msg  import Pose
from project_msgs.msg   import Object, ObjectArray
from detectors.mapping  import DemoNode


# -----------------------------------------------------------------------------
# HSV ranges: [Hmin, Hmax], [Smin, Smax], [Vmin, Vmax]
# OpenCV H is 0-180; S and V are 0-255.
# Blue often needs a wider range (S/V) under real lighting.
# -----------------------------------------------------------------------------
PURPLE_HSV = np.array([
    [0, 3],      # H: purple range
    [35, 62],    # S
    [132, 189],  # V
])
# Blue: widened S and V so dimmer/brighter or less saturated blue still matches
BLUE_HSV = np.array([
    [40, 114],   # H: blue (slightly into cyan and violet)
    [98, 188],   # S: allow lower saturation
    [126, 186],   # V: allow darker and brighter
])


class chaddetectorNode(Node):
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    purple = (255,   0, 255)
    white  = (255, 255, 255)

    def __init__(self, name):
        super().__init__(name)
        self.mapper = DemoNode("mapper")
        self.x0 = -0.505
        self.y0 = 0.76

        #self.create_publisher takes in: (Type, Topic, Queue):

        # Testing something: make a Talker functiion? or try using the terminal 

        
        self.pubrgb   = self.create_publisher(Image, name + '/image_raw', 3)
        self.pubbin   = self.create_publisher(Image, name + '/binary', 3)
        self.pub_bin_purple = self.create_publisher(Image, name + '/binary_purple', 3)
        self.pub_bin_blue   = self.create_publisher(Image, name + '/binary_blue', 3)
        self.pub_objects    = self.create_publisher(ObjectArray, name + '/objects', 3)
        self.pub_purple = self.create_publisher(Point, name + '/purple_bottle', 3)
        self.pub_blue   = self.create_publisher(Point, name + '/blue_bottle', 3)

        self.bridge = cv_bridge.CvBridge()
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Publish at most one bottle (blue or purple) every this many seconds
        self.bottle_publish_interval = 2.0
        self._last_bottle_publish_time = None

        self.get_logger().info("Chad bottle detector running (purple + blue)...")

    def pixel_to_pose(self, image, u, v):
        """Return (x, y) in world frame or None."""
        xy = self.mapper.pixelToWorld(image, u, v, self.x0, self.y0)
        if xy is None:
            return None
        return (-float(xy[0]), float(xy[1])) #FLIPPED THE X SIGN HERE

    def publish_bottle(self, image, u, v, publisher):
        now = self.get_clock().now()
        if self._last_bottle_publish_time is not None:
            elapsed = (now - self._last_bottle_publish_time).nanoseconds * 1e-9
            if elapsed < self.bottle_publish_interval:
                return
        pose_xy = self.pixel_to_pose(image, u, v)
        if pose_xy is not None:
            msg = Point()
            msg.x = pose_xy[0]
            msg.y = pose_xy[1]
            msg.z = 0.0
            publisher.publish(msg)
            self._last_bottle_publish_time = now
            self.get_logger().info("Bottle published")

    def shutdown(self):
        self.destroy_node()

    def _detect_bottle_center(self, binary, min_area=100, max_area=1000):
        """Find largest contour in binary mask; return (u, v) centroid or None."""
        centers_list = []

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # best = max(contours, key=cv2.contourArea)
        # if cv2.contourArea(best) < min_area:
        #     return None
        # M = cv2.moments(best)

        for contour in contours:
            if min_area < cv2.contourArea(contour) < max_area:
                contour_moment = cv2.moments(contour)
                tuple = ((int(contour_moment["m10"] / contour_moment["m00"])), int(contour_moment["m01"] / contour_moment["m00"]))
                centers_list.append(tuple)
            
        # if M["m00"] == 0:
        #     return None
        # u = int(M["m10"] / M["m00"])
        # v = int(M["m01"] / M["m00"])

        if not centers_list:
            return None

        #self.get_logger().info(f"Found {len(centers_list)} contours in binary mask, centers: \n {centers_list}")
        return centers_list

    def process(self, msg):
        assert msg.encoding == "rgb8"
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        H, W = frame.shape[:2]

        # Binary masks for each color
        binary_purple = cv2.inRange(hsv, PURPLE_HSV[:, 0], PURPLE_HSV[:, 1])
        binary_blue   = cv2.inRange(hsv, BLUE_HSV[:, 0], BLUE_HSV[:, 1])

        # # Morphology to clean up
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # binary_purple = cv2.morphologyEx(binary_purple, cv2.MORPH_CLOSE, kernel)
        # binary_purple = cv2.morphologyEx(binary_purple, cv2.MORPH_OPEN, kernel)
        # binary_blue   = cv2.morphologyEx(binary_blue, cv2.MORPH_CLOSE, kernel)
        # binary_blue   = cv2.morphologyEx(binary_blue, cv2.MORPH_OPEN, kernel)
        # # Exclude purple pixels from blue so one bottle doesn't win both
        # binary_blue = cv2.bitwise_and(binary_blue, cv2.bitwise_not(binary_purple))

        # # Combined binary for debug
        # binary_combined = cv2.bitwise_or(binary_purple, binary_blue)
        self.pub_bin_purple.publish(self.bridge.cv2_to_imgmsg(binary_purple, "mono8"))
        self.pub_bin_blue.publish(self.bridge.cv2_to_imgmsg(binary_blue, "mono8"))

        # Detect bottle centers (use centroid of largest blob)
        center_purple = self._detect_bottle_center(binary_purple)
        center_blue   = self._detect_bottle_center(binary_blue)

        # Build ObjectArray: blue first, then purple (order for touch sequence)
        objects = ObjectArray()
        objects.objects = []

        # Draw and publish
        if center_blue is not None:
            # self.get_logger().info("Blue object found, count: " + str(len(center_blue)))
            for i, center in enumerate(center_blue):
                u, v = center
                cv2.circle(frame, (u, v), 25, self.blue, 2)
                cv2.circle(frame, (u, v), 2, self.red, -1)
                cv2.putText(frame, "blue", (u - 20, v - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.blue, 1)
                #self.get_logger().info("pub blue")
                self.publish_bottle(frame, u, v, self.pub_blue)
                xy = self.pixel_to_pose(frame, u, v)
                if xy is not None:
                    obj = Object(type=Object.BOTTLE_BLUE, pose=Pose())
                    obj.pose.position.x = xy[0]
                    obj.pose.position.y = xy[1]
                    obj.pose.position.z = 0.0
                    obj.pose.orientation.w = 1.0
                    objects.objects.append(obj)

        if center_purple is not None:
            #self.get_logger().info("Purple object found, count: " + str(len(center_purple)))
            for i, center in enumerate(center_purple):
                u, v = center
                cv2.circle(frame, (u, v), 25, self.purple, 2)
                cv2.circle(frame, (u, v), 2, self.red, -1)
                cv2.putText(frame, "purple", (u - 30, v - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.purple, 1)
                self.publish_bottle(frame, u, v, self.pub_purple)
                xy = self.pixel_to_pose(frame, u, v)
                if xy is not None:
                    obj = Object(type=Object.BOTTLE_PURPLE, pose=Pose())
                    obj.pose.position.x = xy[0]
                    obj.pose.position.y = xy[1]
                    obj.pose.position.z = 0.0
                    obj.pose.orientation.w = 1.0
                    objects.objects.append(obj)

        self.pub_objects.publish(objects)
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        # self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary_combined, "mono8"))


def main(args=None):
    rclpy.init(args=args)
    node = chaddetectorNode('chaddetector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

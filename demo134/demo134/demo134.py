'''demo134.py

   Demonstration node to interact with the HEBIs.

   Similar to 133a, this publishes joint commands every 10ms.  But the
   topic is now /joint_commands.

   To start the trajectory, the initialization also (temporarily)
   subscribes to /joint_states and remembers the initial joint
   positions.


   Node:        /demo
   Subscribe:   /joint_states (once)    sensor_msgs/JointState
   Publish:     /joint_commands         sensor_msgs/JointState
'''

import rclpy
import numpy as np
import tf2_ros
import sys

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from std_msgs.msg       import Float32MultiArray


#
#   Definitions
#
RATE = 100.0            # Hertz


#
#   DEMO Node Class
#
#   This inherits all the standard ROS node stuff, but adds an
#   update() method to be called regularly by an internal timer and a
#   shutdown method to stop the timer.
#
#   Arguments are the node name and a future object (to force a shutdown).
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        # Create a temporary subscriber to grab the initial position.
        self.position0 = self.grabfbk()
        self.get_logger().info("Initial positions: %r" % self.position0)

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        self.T = 2
        self.qd = []
        self.qddot = []

        ##############################################################
        # Setup the logistics of the node:
        # Add a publisher to send the joint commands.
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Do you need any other subscribers?  Place them here.
        # self.create_subscription(Type, '/topic', callback, 1)
        # self.create_subscription(JointT)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a timer to keep calculating/sending commands.
        rate           = RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(1/rate, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()


    # Grab a single feedback - DO NOT CALL THIS REPEATEDLY!
    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        # Return the values.
        return self.grabpos

    # Send a command.
    def sendcmd(self, pos, vel, eff = []):
        # Build up the message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp    = self.get_clock().now().to_msg()
        cmdmsg.header.frame_id = 'world'
        cmdmsg.name            = ['one', 'two', 'three']
        cmdmsg.position        = pos
        cmdmsg.velocity        = vel
        cmdmsg.effort          = eff
        self.pubcmd.publish(cmdmsg)


    ######################################################################
    # Handlers
    def goto(self, t, T, p0, pf):
        # Cubic time-scaling trajectory (position, velocity)
        tau = min(max(t / T, 0.0), 1.0)
        p = p0 + (pf - p0) * (3 * tau**2 - 2 * tau**3)
        v = (pf - p0) / T * (6 * tau * (1 - tau))
        return (p, v)

    # Timer (100Hz) update.
    def update(self):
        now = self.get_clock().now()
        t = (now - self.starttime).nanoseconds * 1e-9
        
        target1 = [self.position0[0], np.pi/2, self.position0[2]]  # base, shoulder, elbow all straight up
        target2 = [0.0, np.pi/2, np.pi/2]  # base, shoulder, elbow all straight up
        target3 = [np.pi, np.pi/2, np.pi]  # rotate base and elbow
        target4 = [np.pi, np.pi/2.0, 0.0]  # move elbow
        target5 = [np.pi, np.pi, np.pi/2.0]  # go flat


        if t < self.T:
            # Trajectory parameters
            q0 = self.position0
            # Compute desired position and velocity for each joint
            self.qd = []
            self.qddot = []
            for i in range(len(self.position0)):
                pos, vel = self.goto(t, self.T, q0[i], target1[i])
                self.qd.append(pos)
                self.qddot.append(vel)

            self.sendcmd(self.qd, self.qddot)

        elif t < 2*self.T:
            # Trajectory parameters
            q0 = target1

            # Compute desired position and velocity for each joint
            self.qd = []
            self.qddot = []
            for i in range(len(self.position0)):
                pos, vel = self.goto(t-self.T, self.T, q0[i], target2[i])
                self.qd.append(pos)
                self.qddot.append(vel)

            self.sendcmd(self.qd, self.qddot)

        elif t < 3*self.T:
             # Trajectory parameters
            q0 = target2

            # Compute desired position and velocity for each joint
            self.qd = []
            self.qddot = []
            for i in range(len(self.position0)):
                pos, vel = self.goto(t-2*self.T, self.T, q0[i], target3[i])
                self.qd.append(pos)
                self.qddot.append(vel)

            self.sendcmd(self.qd, self.qddot)

        elif t < 4*self.T:
             # Trajectory parameters
            q0 = target3

            # Compute desired position and velocity for each joint
            self.qd = []
            self.qddot = []
            for i in range(len(self.position0)):
                pos, vel = self.goto(t-3*self.T, self.T, q0[i], target4[i])
                self.qd.append(pos)
                self.qddot.append(vel)

            self.sendcmd(self.qd, self.qddot)
        
        else:
             # Trajectory parameters
            q0 = target4

            # Compute desired position and velocity for each joint
            self.qd = []
            self.qddot = []
            for i in range(len(self.position0)):
                pos, vel = self.goto(t-4*self.T, self.T, q0[i], target5[i])
                self.qd.append(pos)
                self.qddot.append(vel)

            self.sendcmd(self.qd, self.qddot)

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Instantiate the DEMO node.
    node = DemoNode('demo', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the node is complete
    # (as signaled by the future object).
    rclpy.spin_until_future_complete(node, future)

    # Report the reason for shutting down.
    if future.done():
        node.get_logger().info("Stopping: " + future.result())
    else:
        node.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

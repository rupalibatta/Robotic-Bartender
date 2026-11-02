'''go_to_zero_spline.py

   Standalone node: goto from current position to [0,0,0,0,0,0] (same goto as touchtable).
   No segments, no trajectory node. Reads /joint_states at start, runs one goto,
   publishes /joint_commands at 100 Hz. No gravity compensation.

   Node:        /go_to_zero_spline
   Subscribe:   /joint_states     sensor_msgs/JointState
   Publish:     /joint_commands   sensor_msgs/JointState (6 joints)
'''

import rclpy
import numpy as np
from asyncio import Future
from rclpy.node import Node
from sensor_msgs.msg import JointState

RATE = 100.0
DT = 1.0 / RATE
T_MOVE = 3.0
ZERO_POSE = np.array([-1.5, -0.5, -0.5, 0.0, 0.0, 0.0], dtype=float) # The point we tell it to move to 


class GoToZeroSplineNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        self.jointnames = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper']
        self.position0 = np.array(self._grabfbk(), dtype=float)
        self.get_logger().info("Initial positions (6): %s" % self.position0.tolist())

        self.starttime = self.get_clock().now()
        self.pcmd = self.position0.copy()
        self.vcmd = np.zeros(6)

        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)
        self.create_subscription(JointState, '/joint_states', self._recvact, 10)

        self.get_logger().info("Waiting for /joint_commands subscriber...")
        while not self.count_subscribers('/joint_commands'):
            pass

        self.timer = self.create_timer(DT, self._update)
        self.get_logger().info("Go-to-zero: moving to 0 0 0 0 0 0 over %.1fs (goto, no segments)" % T_MOVE)
    


    def _grabfbk(self):
        grabpos = [0.0] * 6
        ready = [False]
        def cb(msg):
            grabpos[:] = list(msg.position)
            ready[0] = True
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        while not ready[0]:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)
        return grabpos

    def _recvact(self, msg):
        pass

    def _goto(self, t, T, p0, pf):
        """Same as touchtable.py: smooth step (3*tau^2 - 2*tau^3), zero vel at ends."""
        tau = min(max(t / T, 0.0), 1.0)
        p = p0 + (pf - p0) * (3 * tau**2 - 2 * tau**3)
        v = (pf - p0) / T * (6 * tau * (1 - tau))
        return p, v
    

    def recv_bottle(self, obj):
        if self.busy:
            return
        self.busy = True
        self.strip_tap_state = 0
        dx, dy = self.error_mapping(obj.position.x, obj.position.y)
        x = obj.position.x - dx
        y = obj.position.y - dy
        self.p_touch = np.array([x, y, 0.0], dtype=float)
        self.received_point_bool = True
        self.starttime = self.get_clock().now()
        self.touch_start_tip_pos = None
        self.return_initialized = False
        self.get_logger().info(f"Received bottle at ({x:.3f}, {y:.3f})")
        

    def _update(self):
        t = (self.get_clock().now() - self.starttime).nanoseconds * 1e-9
        if t <= T_MOVE:
            self.pcmd, self.vcmd = self._goto(t, T_MOVE, self.position0, ZERO_POSE)
        else:
            self.pcmd = ZERO_POSE.copy()
            self.vcmd = np.zeros(6)

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'world'
        cmd.name = self.jointnames
        cmd.position = [float(x) for x in self.pcmd.tolist()]
        cmd.velocity = [float(x) for x in self.vcmd.tolist()]
        cmd.effort = [0.0] * 6
        self.pubcmd.publish(cmd)

    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = GoToZeroSplineNode('go_to_zero_spline', future)
    rclpy.spin_until_future_complete(node, future)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()



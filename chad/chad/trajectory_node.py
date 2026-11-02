'''trajectory_node.py

   Chad trajectory node: runs the segment queue and spline, publishes joint_commands.
   Receives segments from the brain (SegmentArray => replace). Publishes Empty for
   TrajectoryComplete when done and Contact when halted.

   Node:        /trajectory
   Subscribe:   /joint_states              sensor_msgs/JointState
                /trajectory/segments       project_msgs/SegmentArray  (replace)
                /trajectory/halt           std_msgs/Empty              (halt + publish Contact)
   Publish:     /joint_commands            sensor_msgs/JointState (6 joints)
                /trajectory/trajectory_complete   std_msgs/Empty
                /trajectory/contact               std_msgs/Empty
'''

import rclpy
import numpy as np
from math import pi, sin, cos

from asyncio import Future
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from project_msgs.msg import SegmentArray

RATE = 100.0
DT = 1.0 / RATE
N_ARM = 3  # base, shoulder, elbow; wrist fixed at 0


class Segment:
    """Waypoint plus time (for trajectory queue)."""
    def __init__(self, pf, vf, Tmove):
        self.pf = list(pf)
        self.vf = list(vf)
        self.Tmove = float(Tmove)


class Spline:
    """Quintic spline from (tcmd, pcmd, vcmd) to segment waypoint."""
    def __init__(self, tcmd, pcmd, vcmd, segment):
        self.t0 = tcmd
        self.T = segment.Tmove
        p0 = np.array(pcmd)
        v0 = np.array(vcmd)
        pf = np.array(segment.pf)
        vf = np.array(segment.vf)
        T = self.T
        self.a = p0
        self.b = v0
        self.c = np.zeros_like(p0)
        self.d = +10 * (pf - p0) / T**3 - 6 * v0 / T**2 - 4 * vf / T**2
        self.e = -15 * (pf - p0) / T**4 + 8 * v0 / T**3 + 7 * vf / T**3
        self.f = +6 * (pf - p0) / T**5 - 3 * v0 / T**4 - 3 * vf / T**4

    def evaluate(self, t):
        tau = t - self.t0
        p = self.a + self.b * tau + self.c * tau**2 + self.d * tau**3 + self.e * tau**4 + self.f * tau**5
        v = self.b + 2 * self.c * tau + 3 * self.d * tau**2 + 4 * self.e * tau**3 + 5 * self.f * tau**4
        return (p.tolist(), v.tolist())


class TrajectoryNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        self.jointnames = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper']
        self.position0 = np.array(self._grabfbk(), dtype=float)
        self.get_logger().info(f"Initial positions (6): {self.position0.tolist()}")

        self.tcmd = 0.0
        self.pcmd = list(self.position0[:N_ARM])
        self.vcmd = [0.0, 0.0, 0.0]
        self.segments = []
        self.spline = None
        self.abort = False
        self.starttime = self.get_clock().now()
        self._trajectory_complete_pending = False
        self._last_idle_log = 0.0

        self.A = -1.8
        self.B = -0.2
        self.actpos = self.position0.copy()
        self.actvel = np.zeros(6)

        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)
        self.pub_complete = self.create_publisher(Empty, '/trajectory/trajectory_complete', 10)
        self.pub_contact = self.create_publisher(Empty, '/trajectory/contact', 10)

        self.create_subscription(JointState, '/joint_states', self._recvact, 10)
        self.create_subscription(SegmentArray, '/trajectory/segments', self._recv_segments, 10)
        self.create_subscription(Empty, '/trajectory/halt', self._recv_halt, 10)

        self.get_logger().info("Waiting for /joint_commands subscriber...")
        while not self.count_subscribers('/joint_commands'):
            pass

        self.timer = self.create_timer(DT, self._update)
        self.get_logger().info("Trajectory node running (segments from brain, TrajectoryComplete/Contact)")

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
        self.actpos = np.array(msg.position, dtype=float)
        self.actvel = np.array(msg.velocity, dtype=float)

    def _recv_segments(self, msg):
        """Replace queue with incoming list of segments."""
        self.segments = []
        for s in msg.segments:
            self.segments.append(Segment(list(s.pf), list(s.vf), s.t_move))
        self.abort = True
        self._trajectory_complete_pending = False
        self.get_logger().info("Received %d segments (replace)" % len(self.segments))

    def _recv_halt(self, msg):
        """Halt: clear queue, abort current spline, publish Contact."""
        self.segments = []
        self.abort = True
        self._trajectory_complete_pending = False
        self.pub_contact.publish(Empty())
        self.get_logger().info("Halt requested; published Contact")

    def _gravity(self, pos):
        tau_shoulder = self.A * sin(pos[1]) + self.B * cos(pos[1])
        return [0.0, tau_shoulder, 0.0, 0.0, 0.0, 0.0]

    def _sendcmd(self, pos, vel, eff=None):
        if eff is None:
            eff = self._gravity(self.actpos)
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'world'
        cmd.name = self.jointnames
        cmd.position = [float(x) for x in pos]
        cmd.velocity = [float(x) for x in vel]
        cmd.effort = [float(x) for x in eff]
        self.pubcmd.publish(cmd)

    def _update(self):
        t = (self.get_clock().now() - self.starttime).nanoseconds * 1e-9

        if self.spline and ((t - self.spline.t0) > self.spline.T or self.abort):
            self.spline = None
            self.abort = False
            if len(self.segments) == 0:
                self._trajectory_complete_pending = True

        if self.spline is None and len(self.segments) > 0:
            seg = self.segments.pop(0)
            self.spline = Spline(self.tcmd, self.pcmd, self.vcmd, seg)
            self.get_logger().info("Starting spline: pf=%s, T=%.2fs" % (seg.pf, seg.Tmove))

        if self.spline:
            self.pcmd, self.vcmd = self.spline.evaluate(t)
        else:
            self.vcmd = [0.0, 0.0, 0.0]
            if self._trajectory_complete_pending:
                self.pub_complete.publish(Empty())
                self._trajectory_complete_pending = False
                self.get_logger().info("Trajectory complete")
            if t - self._last_idle_log >= 5.0:
                self._last_idle_log = t
                self.get_logger().info("Idle: no segments (run chaddetector with both bottles in view for brain to send segments)")

        self.tcmd = t

        q_cmd = list(self.pcmd) + [0.0, 0.0, 0.0]
        v_cmd = list(self.vcmd) + [0.0, 0.0, 0.0]
        self._sendcmd(q_cmd, v_cmd)

    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = TrajectoryNode('trajectory', future)
    rclpy.spin_until_future_complete(node, future)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

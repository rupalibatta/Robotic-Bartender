#!/usr/bin/env python3
"""
gravity.py

  Move the 6-DOF arm to a hardcoded waiting pose, then hold there.
  Sends the same gravity compensation (tau) as touchtable: shoulder only,
  tau_shoulder = A*sin(pos[1]) + B*cos(pos[1]), for tuning A and B on this robot.

  Node:        /gravity
  Subscribe:  /joint_states
  Publish:    /joint_commands  (position, velocity, effort)
"""

import rclpy
import numpy as np
from math import pi, sin, cos

from rclpy.node import Node
from sensor_msgs.msg import JointState

RATE = 100.0
DT = 1.0 / RATE

# Hardcoded waiting pose (same as sixdof): base, shoulder, elbow, wrist1, wrist2, gripper
WAITING_6 = np.array([0.0, pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=float)

# Spline duration to reach waiting (seconds)
T_GO = 3.0


class GravityNode(Node):
    def __init__(self):
        super().__init__("gravity")
        self.jointnames = [
            "base",
            "shoulder",
            "elbow",
            "wrist1",
            "wrist2",
            "gripper",
        ]
        # Same gravity constants as touchtable.py (tune A, B for this robot)
        self.A = -1.8
        self.B = -0.2

        # Initial pose (grabbed once at start)
        self.position0 = np.array(self._grabfbk(), dtype=float)
        self.get_logger().info("Initial positions (6): %s" % self.position0.tolist())

        self.actpos = self.position0.copy()
        self.actvel = np.zeros(6)
        self.acteff = np.zeros(6)

        self.pubcmd = self.create_publisher(JointState, "/joint_commands", 10)
        self.create_subscription(JointState, "/joint_states", self._recvact, 10)

        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while not self.count_subscribers("/joint_commands"):
            pass

        self.starttime = self.get_clock().now()
        self.timer = self.create_timer(DT, self._update)
        self.get_logger().info(
            "Gravity node: spline to waiting in %.1fs, then hold (A=%.2f, B=%.2f)"
            % (T_GO, self.A, self.B)
        )

    def _grabfbk(self):
        out = [0.0] * 6
        done = [False]

        def cb(msg):
            out[:] = list(msg.position)[:6]
            done[0] = True

        sub = self.create_subscription(JointState, "/joint_states", cb, 1)
        while not done[0]:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)
        return out

    def _recvact(self, msg):
        self.actpos = np.array(msg.position, dtype=float)[:6]
        self.actvel = np.array(msg.velocity, dtype=float)[:6]
        self.acteff = np.array(msg.effort, dtype=float)[:6]

    def _goto(self, t, T, p0, pf):
        """Smooth spline: p0 -> pf over time T. Same as touchtable goto."""
        tau = min(max(t / T, 0.0), 1.0)
        p = p0 + (pf - p0) * (3 * tau**2 - 2 * tau**3)
        v = (pf - p0) / T * (6 * tau * (1 - tau))
        return p, v

    def _gravity(self, pos):
        """Same formula as touchtable: shoulder only. Returns 6-vector effort."""
        tau_shoulder = self.A * sin(pos[1]) + self.B * cos(pos[1])
        return np.array([0.0, tau_shoulder, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def _sendcmd(self, pos, vel, eff):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "world"
        cmd.name = self.jointnames
        cmd.position = [float(x) for x in pos]
        cmd.velocity = [float(x) for x in vel]
        cmd.effort = [float(x) for x in eff]
        self.pubcmd.publish(cmd)

    def _update(self):
        now = self.get_clock().now()
        t = (now - self.starttime).nanoseconds * 1e-9

        if t < T_GO:
            q, qdot = self._goto(t, T_GO, self.position0, WAITING_6)
            q_cmd = np.array(q, dtype=float)
            qdot_cmd = np.array(qdot, dtype=float)
        else:
            q_cmd = WAITING_6.copy()
            qdot_cmd = np.zeros(6)

        # Gravity comp from current actual position (same as touchtable)
        tau = self._gravity(self.actpos)
        self._sendcmd(q_cmd, qdot_cmd, tau)


def main(args=None):
    rclpy.init(args=args)
    node = GravityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

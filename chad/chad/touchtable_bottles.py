'''touchtable_bottles.py

   Chad 6-DOF robot: touch blue bottle then slide over to purple bottle.
   Uses ObjectArray from chaddetector. IK uses first 3 joints (base, shoulder,
   elbow) from Chad URDF; wrist1, wrist2, gripper held at 0.
   Trajectory queue: Segment/Spline (quintic). Touch at table level z=0.

   Node:        /touchtable_bottles
   Subscribe:   /joint_states           sensor_msgs/JointState
                /chaddetector/objects   project_msgs/ObjectArray
   Publish:     /joint_commands         sensor_msgs/JointState (6 joints)
'''

import rclpy
import numpy as np
from math import sin, cos, atan2

from asyncio import Future
from rclpy.node import Node
from sensor_msgs.msg import JointState
from project_msgs.msg import Object, ObjectArray

from chad.KinematicChainSol import KinematicChain
from chad.arm_planning_config import (
    ARM_JOINTNAMES,
    CHAIN_BASE_FRAME,
    CHAIN_TIP_FRAME,
    WAITING_ARM,
    T_MOVE,
    IK_TOL,
    IK_MAX_ITERS,
    IK_MU,
    TOUCH_Z,
)


RATE = 100.0
DT = 1.0 / RATE


class Segment:
    """Waypoint plus time to get there (for trajectory queue)."""
    def __init__(self, pf, vf, Tmove):
        self.pf = list(pf)
        self.vf = list(vf)
        self.Tmove = float(Tmove)


class Spline:
    """Quintic spline from (tcmd, pcmd, vcmd) to segment waypoint. Zero accel at ends."""
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


class TouchtableBottlesNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        # 3-DOF arm: chain and waiting pose from shared config (see arm_planning_config.py)
        self.arm_jointnames = list(ARM_JOINTNAMES)
        self.chain = KinematicChain(self, CHAIN_BASE_FRAME, CHAIN_TIP_FRAME, self.arm_jointnames)

        # All 6 joints for /joint_commands (wrist1, wrist2, gripper held at 0)
        self.jointnames = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper']
        self.n_arm = 3

        # Grab initial feedback (6 joints)
        self.position0 = np.array(self._grabfbk(), dtype=float)
        self.get_logger().info(f"Initial positions (6): {self.position0.tolist()}")

        self.waiting_arm = WAITING_ARM.copy()
        self.waiting = np.concatenate([self.waiting_arm, [0.0, 0.0, 0.0]]).astype(float)

        # Trajectory state (arm waypoints are 3-DOF; wrist fixed)
        self.tcmd = 0.0
        self.pcmd = list(self.position0[: self.n_arm])
        self.vcmd = [0.0, 0.0, 0.0]
        self.segments = []
        self.spline = None
        self.abort = False
        self.starttime = self.get_clock().now()

        self.A = -1.8
        self.B = -0.2
        self.actpos = self.position0.copy()
        self.actvel = np.zeros(6)

        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)
        self.create_subscription(JointState, '/joint_states', self._recvact, 10)
        self.create_subscription(ObjectArray, '/chaddetector/objects', self._recv_objects, 10)

        self.get_logger().info("Waiting for /joint_commands subscriber...")
        while not self.count_subscribers('/joint_commands'):
            pass

        self.timer = self.create_timer(DT, self._update)
        self.get_logger().info("Touchtable bottles node running")

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

    def _recv_objects(self, msg):
        if len(msg.objects) < 2:
            return
        blue_pose = None
        purple_pose = None
        for obj in msg.objects:
            if obj.type == Object.BOTTLE_BLUE:
                blue_pose = (obj.pose.position.x, obj.pose.position.y, TOUCH_Z)
            elif obj.type == Object.BOTTLE_PURPLE:
                purple_pose = (obj.pose.position.x, obj.pose.position.y, TOUCH_Z)
        if blue_pose is None or purple_pose is None:
            return
        zero = [0.0, 0.0, 0.0]
        self.segments = [
            Segment(self.waiting_arm, zero, T_MOVE),
        ]
        self.abort = True
        self.get_logger().info("Replaced segments: 1 segment to waiting_arm")

    def _newton_rap(self, p_des, q0=None, tol=IK_TOL, max_iters=IK_MAX_ITERS):
        p_des = np.array(p_des, dtype=float)
        if q0 is None:
            q = self.waiting_arm.copy()
            q[0] = atan2(p_des[1], p_des[0])
        else:
            q = np.array(q0[:3], dtype=float)
        for _ in range(max_iters):
            p, _, Jv, _ = self.chain.fkin(q)
            p = np.array(p, dtype=float)
            Jv = np.array(Jv, dtype=float)
            e = p_des - p
            if np.linalg.norm(e) < tol:
                return q
            dq = Jv.T @ np.linalg.solve(
                Jv @ Jv.T + (IK_MU ** 2) * np.eye(3),
                e,
            )
            q = q + dq
        return None

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

        if self.spline is None and len(self.segments) > 0:
            self.spline = Spline(self.tcmd, self.pcmd, self.vcmd, self.segments.pop(0))

        if self.spline:
            self.pcmd, self.vcmd = self.spline.evaluate(t)
        else:
            self.vcmd = [0.0, 0.0, 0.0]

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
    node = TouchtableBottlesNode('touchtable_bottles', future)
    rclpy.spin_until_future_complete(node, future)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

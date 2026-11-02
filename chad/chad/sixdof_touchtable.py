'''sixdof_touchtable.py

   Chad 6-DOF: touch blue bottle only. Same style as old touchtable (state machine
   in update with goto, no segments). Interfaces with chaddetector: subscribes to
   /chaddetector/objects; when blue bottle is seen, runs sequence
   lift -> waiting -> move to blue -> hold -> return to waiting.
   Purple bottle is not required. No gravity compensation.
   IK uses 3-DOF arm (world->lowerarm); wrist/gripper held at 0.

   Node:        /sixdof_touchtable
   Subscribe:   /joint_states           sensor_msgs/JointState
                /chaddetector/objects   project_msgs/ObjectArray
   Publish:     /joint_commands        sensor_msgs/JointState (6 joints)
'''

import rclpy
import numpy as np
from math import pi, atan2

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
    IK_TOL,
    IK_MAX_ITERS,
    IK_MU,
    TOUCH_Z,
)

RATE = 100.0
DT = 1.0 / RATE


class SixDofTouchtableNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        self.jointnames = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper']
        # Use actual robot position from /joint_states (no rest pose)
        pos0 = self.grabfbk()
        self.position0 = np.array(pos0, dtype=float)
        self.get_logger().info("Using current robot position from /joint_states (6): %s" % self.position0.tolist())

        # Trajectory timing (seconds)
        self.T1 = 2.0   # lift shoulder
        self.T2 = 2.0   # go to waiting
        self.T4 = 2.0   # move to touch point
        self.TH = 0.4   # hold
        self.T5 = 2.0   # return to waiting

        self.waiting = np.array(WAITING_ARM, dtype=float)  # 3-DOF arm
        self.p_touch = np.array([0.5, 0.5, TOUCH_Z], dtype=float)
        self.lam = 12.0
        self.mu = IK_MU
        self.elbow_up_sign = -1.0

        # 3-DOF chain for IK
        self.chain = KinematicChain(self, CHAIN_BASE_FRAME, CHAIN_TIP_FRAME, list(ARM_JOINTNAMES))
        self.q_cmd = self.position0[:3].copy()
        self.qdot_cmd = np.zeros(3)

        self.lift_start_q = None
        self.touch_start_tip_pos = None
        self.return_q0 = None
        self.return_initialized = False

        self.actpos = self.position0.copy()
        self.actvel = np.zeros(6)

        self.busy = False
        self.received_point_bool = False
        self._last_objects_log_t = 0.0

        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)
        self.create_subscription(JointState, '/joint_states', self.recvact, 10)
        self.create_subscription(ObjectArray, '/chaddetector/objects', self.recv_objects, 10)

        self.get_logger().info("Waiting for /joint_commands subscriber...")
        while not self.count_subscribers('/joint_commands'):
            pass

        self.starttime = self.get_clock().now()
        self.timer = self.create_timer(DT, self.update)
        self.get_logger().info("SixDOF touchtable running (chaddetector -> blue bottle only, no gravity)")

    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()

    def recv_objects(self, msg):
        # Log occasionally so we know we're receiving (types: 0=DISK, 1=STRIP, 2=BOTTLE_BLUE, 3=BOTTLE_PURPLE)
        if len(msg.objects) > 0 and not self.busy:
            t = self.get_clock().now().nanoseconds * 1e-9
            if t - self._last_objects_log_t >= 2.0:
                self._last_objects_log_t = t
                types = [o.type for o in msg.objects]
                self.get_logger().info("ObjectArray: %d objects, types=%s" % (len(msg.objects), types))
        if self.busy:
            return
        for obj in msg.objects:
            if obj.type == Object.BOTTLE_BLUE:
                self.get_logger().info("Blue object found")
                self.p_touch = np.array([obj.pose.position.x, obj.pose.position.y, TOUCH_Z], dtype=float)
                self.busy = True
                self.received_point_bool = True
                self.starttime = self.get_clock().now()
                self.touch_start_tip_pos = None
                self.return_initialized = False
                self.lift_start_q = None
                self.get_logger().info("Received blue bottle at (%.2f, %.2f)" % (self.p_touch[0], self.p_touch[1]))
                return

    def grabfbk(self):
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

    def sendcmd(self, pos, vel, eff=None):
        if eff is None:
            eff = [0.0] * 6
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'world'
        cmd.name = self.jointnames
        cmd.position = [float(x) for x in pos]
        cmd.velocity = [float(x) for x in vel]
        cmd.effort = [float(x) for x in eff]
        self.pubcmd.publish(cmd)

    def recvact(self, msg):
        self.actpos = np.array(msg.position, dtype=float)
        self.actvel = np.array(msg.velocity, dtype=float)

    def goto(self, t, T, p0, pf):
        tau = min(max(t / T, 0.0), 1.0)
        p = p0 + (pf - p0) * (3 * tau**2 - 2 * tau**3)
        v = (pf - p0) / T * (6 * tau * (1 - tau))
        return p, v

    def enforce_elbow_up(self, q):
        q = np.array(q, dtype=float)
        if self.elbow_up_sign < 0:
            q[2] = min(q[2], -1e-3)
        else:
            q[2] = max(q[2], 1e-3)
        return q

    def newton_rap(self, p_des, q0=None, tol=IK_TOL, max_iters=IK_MAX_ITERS):
        p_des = np.array(p_des, dtype=float)
        if q0 is None:
            q = self.waiting.copy()
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
            dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + (self.mu**2) * np.eye(3), e)
            q = q + dq
        return None

    def update(self):
        now = self.get_clock().now()
        t = (now - self.starttime).nanoseconds * 1e-9

        t1 = self.T1
        t2 = t1 + self.T2
        t4 = t2 + self.T4
        th = t4 + self.TH
        t5 = th + self.T5
        phase_end = t5

        # Idle: hold current position (6 joints)
        if not self.busy or not self.received_point_bool:
            self.q_cmd = self.actpos[:3].copy()
            self.qdot_cmd = np.zeros(3)
            pos_cmd = list(self.q_cmd) + [0.0, 0.0, 0.0]
            vel_cmd = list(self.qdot_cmd) + [0.0, 0.0, 0.0]
            self.sendcmd(pos_cmd, vel_cmd)
            return

        # Finished sequence (return to waiting done): go idle
        if t > phase_end:
            self.busy = False
            self.get_logger().info("Finished blue bottle touch")
            return

        # ---------- Phase: lift shoulder ----------
        if t < t1:
            if self.lift_start_q is None:
                self.lift_start_q = self.actpos[:3].copy()
            q0 = self.lift_start_q.copy()
            qf = q0.copy()
            qf[1] = 0.0
            q, qdot = self.goto(t, self.T1, q0, qf)
            q[0], q[2] = q0[0], q0[2]
            qdot[0], qdot[2] = 0.0, 0.0
            self.q_cmd = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        # ---------- Phase: go to waiting ----------
        elif t < t2:
            q_start = self.lift_start_q.copy()
            q_start[1] = 0.0
            q, qdot = self.goto(t - t1, self.T2, q_start, self.waiting)
            self.q_cmd = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        # ---------- Phase: Cartesian move to touch ----------
        elif t < t4:
            if self.touch_start_tip_pos is None:
                p0, _, _, _ = self.chain.fkin(self.q_cmd)
                self.touch_start_tip_pos = np.array(p0)
            ptip, _, Jv, _ = self.chain.fkin(self.q_cmd)
            ptip = np.array(ptip)
            Jv = np.array(Jv)
            pc, vc = self.goto(t - t2, self.T4, self.touch_start_tip_pos, self.p_touch)
            ep = pc - ptip
            vr = vc + self.lam * ep
            I = np.eye(3)
            A = Jv @ Jv.T + (self.mu**2) * I
            y = np.linalg.solve(A, vr)
            qdot = Jv.T @ y
            self.q_cmd = self.enforce_elbow_up(self.q_cmd + DT * qdot)
            self.qdot_cmd = qdot

        # ---------- Phase: hold ----------
        elif t < th:
            self.qdot_cmd = np.zeros(3)

        # ---------- Phase: return to waiting ----------
        elif t < t5:
            if not self.return_initialized:
                self.return_initialized = True
                self.return_q0 = self.q_cmd.copy()
            q, qdot = self.goto(t - th, self.T5, self.return_q0, self.waiting)
            self.q_cmd = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        else:
            self.qdot_cmd = np.zeros(3)

        pos_cmd = list(self.q_cmd) + [0.0, 0.0, 0.0]
        vel_cmd = list(self.qdot_cmd) + [0.0, 0.0, 0.0]
        self.sendcmd(pos_cmd, vel_cmd)


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = SixDofTouchtableNode('sixdof_touchtable', future)
    rclpy.spin_until_future_complete(node, future)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

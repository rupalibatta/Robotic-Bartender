'''brain_node.py

   Chad brain node: sees detected objects, plans segment list, sends to trajectory node.
   Subscribes to ObjectArray; when both bottles present, runs IK and publishes SegmentArray
   (replace). Optionally subscribes to trajectory_complete to allow replanning.

   Node:        /brain
   Subscribe:   /chaddetector/objects     project_msgs/ObjectArray
                /trajectory/trajectory_complete   std_msgs/Empty  (optional, to allow replan)
   Publish:     /trajectory/segments      project_msgs/SegmentArray
'''

import rclpy
import numpy as np
from math import atan2

from asyncio import Future
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from project_msgs.msg import Object, ObjectArray, Segment, SegmentArray

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


class BrainNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        # 3-DOF arm: chain and waiting pose from shared config (see arm_planning_config.py)
        self.arm_jointnames = list(ARM_JOINTNAMES)
        self.chain = KinematicChain(self, CHAIN_BASE_FRAME, CHAIN_TIP_FRAME, self.arm_jointnames)
        self.waiting_arm = WAITING_ARM.copy()
        self.busy = False
        self.actpos = np.concatenate([self.waiting_arm, [0.0, 0.0, 0.0]]).astype(float)

        self.pub_segments = self.create_publisher(SegmentArray, '/trajectory/segments', 10)
        self.create_subscription(ObjectArray, '/chaddetector/objects', self._recv_objects, 10)
        self.create_subscription(Empty, '/trajectory/trajectory_complete', self._recv_complete, 10)
        self.create_subscription(JointState, '/joint_states', self._recv_joints, 10)

        self.get_logger().info("Brain node running (ObjectArray -> IK -> SegmentArray)")

    def _recv_joints(self, msg):
        self.actpos = np.array(msg.position, dtype=float)

    def _recv_complete(self, msg):
        self.busy = False
        self.get_logger().info("Trajectory complete; ready for next plan")

    def _recv_objects(self, msg):
        n = len(msg.objects)
        if self.busy:
            self.get_logger().debug("ObjectArray: n=%d, skipping (busy)" % n)
            return
        if n < 2:
            self.get_logger().debug("ObjectArray: n=%d, need 2 objects (blue + purple)" % n)
            return
        blue_pose = None
        purple_pose = None
        for obj in msg.objects:
            if obj.type == Object.BOTTLE_BLUE:
                blue_pose = (obj.pose.position.x, obj.pose.position.y, TOUCH_Z)
            elif obj.type == Object.BOTTLE_PURPLE:
                purple_pose = (obj.pose.position.x, obj.pose.position.y, TOUCH_Z)
        if blue_pose is None or purple_pose is None:
            self.get_logger().info("ObjectArray: n=%d, missing blue=%s or purple=%s" % (n, blue_pose is not None, purple_pose is not None))
            return
        self.busy = True

        # For now: only move to waiting (elbow up). Full blue->purple touch sequence commented out.
        zero = [0.0, 0.0, 0.0]
        segments = [
            (self.waiting_arm.tolist(), zero, T_MOVE),
        ]
        arr = SegmentArray()
        arr.segments = []
        for pf, vf, T in segments:
            s = Segment()
            s.pf = pf
            s.vf = vf
            s.t_move = T
            arr.segments.append(s)
        self.pub_segments.publish(arr)
        self.get_logger().info("Published 1 segment: waiting_arm=%s (base, shoulder, elbow)" % self.waiting_arm.tolist())

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
            dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + (IK_MU ** 2) * np.eye(3), e)
            q = q + dq
        return None

    def shutdown(self):
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = BrainNode('brain', future)
    rclpy.spin_until_future_complete(node, future)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

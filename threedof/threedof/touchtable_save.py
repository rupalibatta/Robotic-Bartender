'''touchtable.py

   Skeleton node to interact with the HEBIs.

   Similar to 133a, this publishes joint commands every 10ms.  But the
   topic is now /joint_commands.  Relative to last week, it also
   listens to /joint_states, so it can compute the gravity commands
   and detect collisions.

   To start the trajectory, the initialization also (temporarily)
   subscribes to /joint_states and remembers the initial joint
   positions.


   Node:        /demo
   Subscribe:   /joint_states           sensor_msgs/JointState
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
from threedof.KinematicChainSol import KinematicChain

from geometry_msgs.msg import Point, Pose


#
#   Definitions
#
RATE = 100.0            # Hertz
DT = 1.0/RATE


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
        self.position0 = np.array(self.grabfbk(), dtype=float)
        self.get_logger().info(f"Initial positions: {self.position0.tolist()}")

        ##############################################################
        # INITIALIZE YOUR DATA!

        # Trajectory timing (seconds)
        self.T1 = 2.0   # lift shoulder
        self.T2 = 2.0   # go to waiting
        self.T4 = 2.0   # move to touch point
        self.TH = 0.4   # hold
        self.T5 = 2.0   # return to waiting

        # Waiting pose (elbow-up)
        self.waiting = np.array([0.0, 0.0, -pi/2], dtype=float)

        # Table / target
        # self.table_clearance = 0.005
        self.p_touch = np.array([0.50, 0.50, 0.0], dtype=float)

        # Convergence bandwidth and damping 
        self.lam = 12.0
        self.mu  = 0.02

        # Elbow-up sign
        self.elbow_up_sign = -1.0

        # ROS:
        self.jointnames = ['base', 'shoulder', 'elbow']
        self.chain = KinematicChain(self, 'world', 'tip', self.jointnames)

        # Command for where to go next
        self.q_cmd    = self.position0.copy()
        self.qdot_cmd = np.zeros(3)

        # Starting position
        self.lift_start_q = None
        self.touch_start_tip_pos = None

        # Ending position
        self.return_q0 = None
        self.return_initialized = False

        # Gravity constants (A and B)
        self.A = -1.8 
        self.B = -0.2

        self.actvel = np.zeros(3, dtype=float)
        self.acteff = np.zeros(3, dtype=float)

        # Contact detection thresholds
        self.contact_pos_thresh = np.array([0.35, 0.35, 0.35]) # rad error per joint
        self.contact_vel_thresh = np.array([1.5, 1.5, 1.5]) # rad/s error per joint
        self.contact_tau_thresh = np.array([2.0, 2.0, 2.0]) # Nm residual per joint

        self.contact_hold_t = 1.0 # hold for 1 second when arm contacts something
        self.contact_hold_start_t = None # time in seconds that the arm started holding (when t 
        # reaches contact_hold_start_t + contact_hold_t the arm will return to waiting position)
        self.contact_printed = False
        self.contact_ignore = False

        self.contact_return = False
        self.contact_return_start_t = None
        self.contact_return_q0 = None

        self.busy = False
        self.strip_tap_state = 0

        # Debounce
        # self.contact_debounce_N = 8
        # self.contact_counter = 0
        # self.contact_hold = False
        # self.contact_logged = False

        ##############################################################
        # Setup the logistics of the node:
        # Add a publisher to send the joint commands.
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Create a subscriber to continually receive joint state messages.
        self.actpos = self.position0.copy()
        self.create_subscription(JointState, '/joint_states', self.recvact, 10)

        self.create_subscription(Point, '/balldetector/disk', self.recv_disk, 10)
        self.create_subscription(Pose, 'balldetector/strip', self.recv_strip, 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a timer to keep calculating/sending commands.
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(DT, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, RATE))
        # Receive /point commands
        self.create_subscription(Point, '/point', self.recvpoint, 10)
        self.received_point_bool = False

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    def recv_disk(self, msg):
        if self.busy:
            return
        self.busy = True
        self.strip_tap_state = 0
        self.p_touch = np.array([msg.x, msg.y, 0.0], dtype=float)
        self.received_point_bool = True
        self.starttime = self.get_clock().now()
        self.touch_start_tip_pos = None
        self.return_initialized = False
        self.get_logger().info(f"Received disk at ({msg.x:.3f}, {msg.y:.3f})")

    # tap both sides
    def recv_strip(self, msg):
        if self.busy:
            return
        self.busy = True
        x, y = msg.position.x, msg.position.y
        theta = 2 * np.arctan2(msg.orientation.z, msg.orientation.w)
        offset = 0.06
        dx = offset * np.cos(theta)
        dy = offset * np.sin(theta)
        self.strip_params = {
            'center': np.array([x, y, 0.0], dtype=float),
            'pt1': np.array([x + dx, y + dy, 0.0], dtype=float),
            'pt2': np.array([x - dx, y - dy, 0.0], dtype=float),
            'theta': theta
        }
        self.strip_tap_state = 1  # 1 = tap first side, 2 = tap second side, 0 = done
        self.p_touch = self.strip_params['pt1']
        self.received_point_bool = True
        self.starttime = self.get_clock().now()
        self.touch_start_tip_pos = None
        self.return_initialized = False
        self.get_logger().info(f"Received strip at ({x:.3f}, {y:.3f}), theta={theta:.2f}")

    
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
        cmdmsg.name            = self.jointnames
        cmdmsg.position        = [float(x) for x in pos]
        cmdmsg.velocity        = [float(x) for x in vel]
        cmdmsg.effort          = [float(x) for x in eff]
        self.pubcmd.publish(cmdmsg)


    ######################################################################
    # Handlers

    # Receive actual state - called repeatedly by incoming messages.
    def recvact(self, msg):
        # Save the actual position.
        self.actpos = np.array(msg.position, dtype=float)
        self.actvel = np.array(msg.velocity, dtype=float)
        self.acteff = np.array(msg.effort, dtype=float)

    def goto(self, t, T, p0, pf):
        tau = min(max(t / T, 0.0), 1.0)
        p = p0 + (pf - p0) * (3*tau**2 - 2*tau**3)
        v = (pf - p0) / T * (6*tau*(1 - tau))
        return p, v
    
    def enforce_elbow_up(self, q):
        q = np.array(q, dtype=float)
        if self.elbow_up_sign < 0:
            q[2] = min(q[2], -1e-3)
        else:
            q[2] = max(q[2],  1e-3)
        return q
    
    def recvpoint(self, pointmsg):
        self.p_touch = np.array([pointmsg.x,
                                 pointmsg.y,
                                 pointmsg.z], dtype=float)
        self.received_point_bool = True

        # Reset timing to start a new motion
        self.starttime = self.get_clock().now()
        self.touch_start_tip_pos = None
        self.return_initialized = False

        self.get_logger().info(
            "Received point %.3f %.3f %.3f" %
            (pointmsg.x, pointmsg.y, pointmsg.z)
        )

    def gravity(self, pos):
        tau_shoulder = self.A * sin(pos[1]) + self.B * cos(pos[1])
        return [0, tau_shoulder, 0]
    
    def contact_detection(self, tau):
        pos_err = np.abs(self.q_cmd - self.actpos)
        vel_err = np.abs(self.qdot_cmd - self.actvel)
        eff_err = np.abs(self.acteff - tau)

        pos_event = np.any(pos_err > self.contact_pos_thresh)
        vel_event = np.any(vel_err > self.contact_vel_thresh)
        eff_event = np.any(eff_err > self.contact_tau_thresh)

        event = pos_event or vel_event or eff_event

        if event:
            self.contact_event_printed = True
            self.get_logger().info(
                f"Contact detected."
                f"position error: {pos_err}"
                f"velocity error: {vel_err}"
                f"efficient error: {eff_err}"
            )

        if not event:
            self.contact_event_printed = False

        return event

        """
                made_contact=False
        if abs(self.actpos - self.qd) > self.contact_threshold:
            made_contact = True
        elif abs(actvel-self.qdot) > self.contact_threshold:
            made_contact = True
        elif abs(acteff - self.tau) > self.contact_threshold:
            made_contact = True
        if made_contact:
            self.get_logger().info("Contact detected. Stopping...")
            self.mode = Mode.STOP
        """

    # Timer (100Hz) update.
    def update(self):
        # Grab the current time.
        now = self.get_clock().now()
        t   = (now - self.starttime).nanoseconds * 1e-9

        t1 = self.T1
        t2 = t1 + self.T2
        t4 = t2 + self.T4
        th = t4 + self.TH # hold time
        t5 = th + self.T5

        '''t_contact_return = 2.0
        
        if (self.contact_hold_start_t is not None):
            hold_t = t-self.contact_hold_start_t
            self.q_cmd = self.actpos.copy()
            self.qdot_cmd = np.zeros(3)
            tau = self.gravity(self.actpos) 
            self.sendcmd(self.q_cmd, self.qdot_cmd, tau)
            if (hold_t >= self.contact_hold_t):
                self.contact_hold_start_t = None
                self.contact_return = True
                self.contact_return_start_t = t
                self.contact_return_q0 = self.actpos.copy()
            return

        if self.contact_return:
            q, qdot = self.goto(t - self.contact_return_start_t, t_contact_return, self.contact_return_q0, self.waiting)
            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot
            tau = self.gravity(self.actpos) 
            self.sendcmd(self.q_cmd, self.qdot_cmd, tau)
            if (t >= t_contact_return):
                self.contact_return = False
                self.contact_ignore = False
                self.starttime = self.get_clock().now()
                self.lift_start_q = None
                self.touch_start_tip_pos = None
                self.return_initialized = False
                if self.strip_tap_state == 2:
                    self.strip_tap_state = 0
                    self.busy = False
            return

        if not self.busy:
            return
            
        '''
        '''
        # Disk tap or strip tap (first or second side)
        if self.strip_tap_state == 0:
            pass
        elif self.strip_tap_state == 1:
            # first side of strip
            if not self.received_point_bool:
                return
            # second side
            if t > self.T1 + self.T2 + self.T4 + self.TH + self.T5:
                self.strip_tap_state = 2
                self.p_touch = self.strip_params['pt2']
                self.received_point_bool = True
                self.starttime = self.get_clock().now()
                self.touch_start_tip_pos = None
                self.return_initialized = False
                self.get_logger().info("Moving to tap second side of strip")
                return
        elif self.strip_tap_state == 2:
            # second side of strip
            if not self.received_point_bool:
                return
            if t > self.T1 + self.T2 + self.T4 + self.TH + self.T5:
                self.strip_tap_state = 0
                self.busy = False
                self.get_logger().info("Finished tapping both sides of strip")
                return
                
       '''
         # Lift shoulder
        if t < t1:
            if self.lift_start_q is None:
                self.lift_start_q = self.actpos.copy()

            q0 = self.lift_start_q.copy()
            qf = q0.copy()
            qf[1] = 0.0

            q, qdot = self.goto(t, self.T1, q0, qf)
            q[0], q[2] = q0[0], q0[2]
            qdot[0], qdot[2] = 0.0, 0.0

            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        # Move to waiting
        elif t < t2:
            q_start = self.lift_start_q.copy()
            q_start[1] = 0.0

            q, qdot = self.goto(t - t1, self.T2, q_start, self.waiting)
            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        # Cartesian move to touch
        elif t < t4:
            if self.touch_start_tip_pos is None:
                p0, _, _, _ = self.chain.fkin(self.q_cmd)
                self.touch_start_tip_pos = np.array(p0)

            ptip, _, Jv, _ = self.chain.fkin(self.q_cmd)
            ptip = np.array(ptip)
            Jv   = np.array(Jv)

            pc, vc = self.goto(t - t2, self.T4,
                               self.touch_start_tip_pos,
                               self.p_touch)

            # pc[2] = max(pc[2], self.table_clearance)
            #print(pc[2])
            ep = pc - ptip
            vr = vc + self.lam * ep

            I = np.eye(3)
            A = Jv @ Jv.T + (self.mu**2) * I
            y = np.linalg.solve(A, vr)
            qdot = Jv.T @ y

            self.q_cmd    = self.enforce_elbow_up(self.q_cmd + DT * qdot)
            self.qdot_cmd = qdot

        # Hold
        elif t < th:
            self.qdot_cmd = np.zeros(3)

        # Return to waiting
        elif t < t5:
            if not self.return_initialized:
                self.return_initialized = True
                self.return_q0 = self.q_cmd.copy()

            q, qdot = self.goto(t - th, self.T5,
                                self.return_q0,
                                self.waiting)
            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        else:
            self.qdot_cmd = np.zeros(3)

        tau = self.gravity(self.actpos) 
        event = False
        if not self.contact_ignore:
            event = self.contact_detection(tau)

        if event and (not self.contact_ignore):
            self.get_logger().info("CONTACT EVENT. Stopping, holding 1s, then will return to waiting position.")
            self.contact_ignore = True
            self.contact_hold_start_t = t
            self.q_cmd = self.actpos.copy()
            self.qdot_cmd = np.zeros(3)
        
        # Send.
        self.sendcmd(self.q_cmd, self.qdot_cmd, tau)


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

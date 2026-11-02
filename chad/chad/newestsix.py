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
from chad.KinematicChainSol import KinematicChain

from enum import Enum, unique

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

class states(Enum):
    WAITING = 0
    GOING_TO_BOTTLE = 1
    GOING_TO_WAITING = 2
    GOING_TO_POINT = 3
    CONTACT_DETECTED = 4
    CLOSING_GRIPPER = 5
    POUR = 6
    GOING_TO_POUR_POSITION = 7

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

        self.state = states.GOING_TO_WAITING

        # Trajectory timing (seconds)
        self.T1 = 2.0   # lift shoulder
        self.T2 = 2.0   # go to waiting
        self.T4 = 2.0   # move to touch point
        self.TH = 0.4   # hold
        self.T5 = 2.0   # return to waiting
        self.T_GRIP = 0.4   # hold at bottle with gripper closed before return

        self.t_start_return = 0.0
        self.t_start_bottle = 0.0
        self.t_close_start = None   # when we entered CLOSING_GRIPPER

        self.t_going_to_waiting = 7.0 # time for GOING_TO_WAITING (and bottle approach duration)

        # Pour position (world x, y, z) and move duration
        self.pour_position = np.array([0.424, 0.301, 0.1], dtype=float)
        self.t_start_pour_pos = None  # when we started moving to pour position
        self.returning_from_pour = False  # True when GOING_TO_WAITING is after pour

        # Pour: wrist2 turn 90° left then back right (seconds)
        self.T_POUR_HOLD  = 0.5   # hold at pour position before wrist motion
        self.T_POUR_TURN  = 1.0   # time for each 90° wrist2 turn
        self.t_pour_start = None  # set when entering POUR
        self.wrist2_rest  = 0.0   # wrist2 angle at pour position (rad)

        # Waiting pose (elbow-up)
        self.waiting = np.array([0.0, 0.0, -1.571, 0.0, 0.0], dtype=float)

        # Table / target
        # self.table_clearance = 0.005
        self.p_touch = np.array([0.5, 0.50, 0.0], dtype=float)
        self.pending_bottle = None

        # Convergence bandwidth and damping 
        self.lam = 12.0
        self.mu  = 0.02

        # Elbow-up sign
        self.elbow_up_sign = 1.0

        # ROS:
        self.jointnames = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'gripper']
        self.jointnames5 = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2']
        self.chain = KinematicChain(self, 'world', 'tip', self.jointnames5)

        # Gripper
        self.gripper_open = 0 # SHOULD BE -0
        self.gripper_closed = -0.45


        # Command for where to go next
        self.q_cmd    = self.position0.copy()[:5]
        self.qdot_cmd = np.zeros(5)

        # Starting position
        self.lift_start_q = None
        self.touch_start_tip_pos = None
        self.wrist1_start = None   # wrist1 at start of bottle approach (for spline)
        self.wrist1_target = None # wrist1 target from wrist1_angle (for spline)

        # Ending position
        self.return_q0 = None
        self.return_initialized = False

        # Gravity constants: shoulder (A,B), elbow (C,D), wrist1 (E,F)
        self.A = 0.4
        self.B = 7
        self.C = 0.1
        self.D = 4.4
        self.E = 0.2
        self.F = 0.3

        self.actvel = np.zeros(6, dtype=float)
        self.acteff = np.zeros(6, dtype=float)

        # Contact detection thresholds
        self.contact_pos_thresh = np.array([0.35, 0.35, 0.35, 0.35, 0.35]) # rad error per joint
        self.contact_vel_thresh = np.array([1.5, 1.5, 1.5, 1.5, 1.5]) # rad/s error per joint
        self.contact_tau_thresh = np.array([2.0, 2.0, 2.0, 2.0, 2.0]) # Nm residual per joint

        self.contact_hold_t = 1.0 # hold for 1 second when arm contacts something
        self.contact_hold_start_t = None # time in seconds that the arm started holding (when t 
        # reaches contact_hold_start_t + contact_hold_t the arm will return to waiting position)
        self.contact_printed = False
        self.contact_ignore = True

        self.contact_return = False
        self.contact_return_start_t = None
        self.contact_return_q0 = None

        self.busy = False
        self.strip_tap_state = 0

        ##############################################################
        # Setup the logistics of the node:
        # Add a publisher to send the joint commands.
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Create a subscriber to continually receive joint state messages.
        self.actpos = self.position0.copy()[:5]
        self.create_subscription(JointState, '/joint_states', self.recvact, 10)

        #self.create_subscription(Pose, 'balldetector/strip', self.recv_strip, 10)
        self.create_subscription(Point, '/chaddetector/blue_bottle', self.recv_bottle, 10)
        #self.create_subscription(Point, '/chaddetector/purple_bottle', self.recv_bottle, 10)

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
        # self.create_subscription(Point, '/point', self.recvpoint, 10)
        self.received_point_bool = False

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    def error_mapping(self, x, y):
        dx = 0.00833 - 0.01313*x - 0.03058*y
        dy = 0.01718 + 0.01973*x - 0.01905*y
        return dx, dy

    def wrist1_angle(self, xb, yb):
        """Return wrist1 angle (rad) for bottle at (xb, yb) in cm."""
        xr, yr = 4.6, 68.2  # origin of robot (cm)
        d = np.sqrt((xb - xr) ** 2 + (yb - yr) ** 2)
        theta = (d - 95.4) / 33.7  # line of best fit
        return theta

    # tap both sides
    def recv_strip(self, msg):
        if self.busy:
            return
        self.busy = True

        x, y = msg.position.x, msg.position.y
        dx, dy = self.error_mapping(x, y)
        x = x - dx
        y = y - dy

        theta = 2 * np.arctan2(msg.orientation.z, msg.orientation.w)
        offset = 0.0575
        nx = -np.sin(theta)
        ny = np.cos(theta)
        self.strip_params = {
            'center': np.array([x, y, 0.0], dtype=float),
            'pt1': np.array([x + offset*nx, y + offset*ny, 0.0], dtype=float),
            'pt2': np.array([x - offset*nx, y - offset*ny, 0.0], dtype=float),
            'theta': theta
        }

    # Receive blue bottle and grab it 
    def recv_bottle(self, msg):
        if self.busy:
            # self.get_logger().info("Saw a bottle but am too busy")
            return
        if self.state == states.POUR:
            # self.get_logger().info("Saw a bottle but am busy pouring")
            return  # ignore new bottle while in pour phase
        self.busy = True

        x, y = msg.x, msg.y
        # dx, dy = self.error_mapping(x, y)
        # x = x - dx
        # y = y - dy

        # self.strip_tap_state = 1  # 1 = tap first side, 2 = tap second side, 0 = done
        # self.p_touch = self.strip_params['pt1']
        # self.received_point_bool = True
        # self.starttime = self.get_clock().now()
        # self.touch_start_tip_pos = None
        # self.return_initialized = False

        z = 8 + 2

        z = z / 100


        self.pending_bottle= np.array([x, y, z], dtype=float)
        # self.pending_bottle= np.array([x, y, 0.06], dtype=float)
        if self.state == states.WAITING:
            self.busy = True
            self.p_touch = self.pending_bottle
            self.pending_bottle = None

            self.starttime = self.get_clock().now()
            self.touch_start_tip_pos = None
            self.t_start_bottle = 0.0

            self.state = states.GOING_TO_BOTTLE
            self.get_logger().info(f"going to BOTTLE at ({x:.3f}, {y:.3f})")
        else:
            # Keep going to waiting; well go to the bottle once waiting is reached.
            self.get_logger().info(
                f"Saw bottle at ({x:.3f}, {y:.3f}) while in {self.state}. "
                f"Pending bottle until we are in waiting position."
            )

    
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
        if self.elbow_up_sign > 0:
            q[2] = min(q[2], -1e-3)
        else:
            q[2] = max(q[2],  1e-3)
        return q
    
    def recvpoint(self, pointmsg):
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z

        dx, dy = self.error_mapping(x, y)

        self.p_touch = np.array([x - dx,
                                 y - dy,
                                 z], dtype=float)
        
        self.received_point_bool = True

        # Reset timing to start a new motion
        self.starttime = self.get_clock().now()
        self.touch_start_tip_pos = None
        self.return_initialized = False

        self.get_logger().info(
            "Received corrected point %.3f %.3f %.3f" %
            (self.p_touch[0],self.p_touch[1], self.p_touch[2])
        )

    def gravity(self, pos):
        ''' tau_wrist1 = - self.E * sin(pos[1] + pos[2] + pos[3]) - self.F * cos(pos[1] + pos[2] + pos[3])
        tau_elbow = tau_wrist1 + self.C * sin(pos[1] + pos[2]) + self.D * cos(pos[1] + pos[2])
        tau_shoulder = tau_elbow + self.A * sin(pos[1]) + self.B * cos(pos[1])
        return [0, tau_shoulder, tau_elbow, tau_wrist1, 0, 0]
        '''
        qs = pos[1]
        qe = pos[2]
        qw = pos[3]

        theta_s = qs + 1.57
        theta_e = theta_s + (qe - 0.02)
        theta_w = theta_e + qw

        tau_wrist1 = -self.E * sin(theta_w) - self.F * cos(theta_w)
        tau_elbow = tau_wrist1 + self.C * sin(theta_e) + self.D * cos(theta_e)
        tau_shoulder = tau_elbow + self.A * sin(theta_s) + self.B * cos(theta_s)

        return [0, tau_shoulder, tau_elbow, tau_wrist1, 0, 0]


    # def contact_detection(self): #tau):
    #     pos_err = np.abs(self.q_cmd - self.actpos)
    #     vel_err = np.abs(self.qdot_cmd - self.actvel)
    #     eff_err = np.abs(self.acteff - self.gravity(self.act)) #tau was the second argument here 

    #     pos_event = np.any(pos_err > self.contact_pos_thresh)
    #     vel_event = np.any(vel_err > self.contact_vel_thresh)
    #     eff_event = np.any(eff_err > self.contact_tau_thresh)

    #     event = pos_event or vel_event or eff_event

    #     if event:
    #         self.contact_event_printed = True
    #         self.get_logger().info(
    #             f"Contact detected."
    #             f"position error: {pos_err}"
    #             f"velocity error: {vel_err}"
    #             f"efficient error: {eff_err}"
    #         )

    #     if not event:
    #         self.contact_event_printed = False

    #     return event


    # Timer (100Hz) update.
    def update(self):
        # Grab the current time.
        now = self.get_clock().now()
        t   = (now - self.starttime).nanoseconds * 1e-9

        # Debug: tip position from actual joint state (x, y, z in world), ~10 Hz
        if len(self.actpos) >= 5:
            ptip_act, _, _, _ = self.chain.fkin(self.actpos[:5])
           # if int(t * 10) != int((t - DT) * 10):
               # self.get_logger().info(
                 #   "tip_xyz: %.4f %.4f %.4f" % (float(ptip_act[0]), float(ptip_act[1]), float(ptip_act[2]))
               # )

        t1 = self.T1
        t2 = t1 + self.T2
        t4 = t2 + self.T4
        th = t4 + self.TH # hold time
        t5 = th + self.T5

        t_contact_return = 2.0

        if self.state == states.GOING_TO_WAITING:
            if self.lift_start_q is None:
                self.lift_start_q = self.actpos.copy()[:5]
                self.t_start_return = t

            q0 = self.lift_start_q.copy()
            q, qdot = self.goto(t - self.t_start_return, self.t_going_to_waiting, q0, self.waiting)
            # self.get_logger().info(f"q0: {q0}, t: {t}, t_start_return: {self.t_start_return}, t_going_to_waiting: {self.t_going_to_waiting}")
            # Note: im not sure if htis should be t-self.t_start_return or just t
            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot
            #self.get_logger().info(f"Line 361 positions: {self.q_cmd.tolist()} vs. actual: {self.actpos[:5].tolist()}")
            
            if (t - self.t_start_return) >= self.t_going_to_waiting:
                self.busy = False
                self.get_logger().info("Reached waiting pose.")
                self.get_logger().info("Return done: pending_bottle=%s, returning_from_pour=%s" % (self.pending_bottle is not None, self.returning_from_pour))

                if self.returning_from_pour:
                    self.state = states.WAITING
                    self.returning_from_pour = False
                    self.get_logger().info("Back at waiting after pour.")
                elif self.pending_bottle is not None:
                    self.busy = True
                    self.p_touch = self.pending_bottle
                    self.pending_bottle = None

                    self.starttime = self.get_clock().now()
                    self.touch_start_tip_pos = None
                    self.t_start_bottle = 0.0

                    self.state = states.GOING_TO_BOTTLE
                    self.get_logger().info(f"Leaving waiting. Going to queued bottle at {self.p_touch}")
                else:
                    self.busy = True
                    self.state = states.GOING_TO_POUR_POSITION
                    self.touch_start_tip_pos = None  # set on first tick
                    self.t_start_pour_pos = None
                    self.get_logger().info("Going to pour position at %s" % self.pour_position.tolist())

        elif self.state == states.GOING_TO_POUR_POSITION:
            if self.touch_start_tip_pos is None:
                p0, _, _, _ = self.chain.fkin(self.q_cmd[:5])
                self.touch_start_tip_pos = np.array(p0)
                self.t_start_pour_pos = t
                self.get_logger().info("Splining to pour position: %s" % self.pour_position.tolist())

            ptip, _, Jv, _ = self.chain.fkin(self.q_cmd[:5])
            ptip = np.array(ptip)
            Jv   = np.array(Jv)
            pc, vc = self.goto(t - self.t_start_pour_pos, self.t_going_to_waiting,
                              self.touch_start_tip_pos, self.pour_position)
            ep = pc - ptip
            vr = vc + self.lam * ep
            I = np.eye(3)
            A = Jv @ Jv.T + (self.mu**2) * I
            y = np.linalg.solve(A, vr)
            qdot = Jv.T @ y
            self.q_cmd    = self.enforce_elbow_up(self.q_cmd[:5] + DT * qdot[:5])
            self.qdot_cmd = qdot[:5]

            if (t - self.t_start_pour_pos) >= self.t_going_to_waiting:
                self.state = states.POUR
                self.t_pour_start = None
                self.get_logger().info("At pour position, starting pour.")

        elif self.state == states.POUR:
            # Hold current pose at pour position (joints 0-3); animate only wrist2
            if self.t_pour_start is None:
                self.t_pour_start = t
                self.wrist2_rest = float(self.q_cmd[4])
                self.pour_pose = self.q_cmd[:5].copy()  # keep arm at pour position
                self.get_logger().info("POUR: started (wrist2_rest=%.3f)" % self.wrist2_rest)

            self.q_cmd[:4] = self.pour_pose[:4]  # hold base, shoulder, elbow, wrist1 at pour pose
            elapsed = t - self.t_pour_start
            if elapsed < self.T_POUR_HOLD:
                wrist2 = self.wrist2_rest
                if int(elapsed * 2) != int((elapsed - DT) * 2):
                    self.get_logger().info("POUR: phase=hold (%.1fs)" % elapsed)
            elif elapsed < self.T_POUR_HOLD + self.T_POUR_TURN:
                tau = min(max((elapsed - self.T_POUR_HOLD) / self.T_POUR_TURN, 0.0), 1.0)
                spline = 3.0 * tau * tau - 2.0 * tau * tau * tau
                wrist2 = self.wrist2_rest + (pi) * spline  # turn 180 left
                if int(elapsed * 2) != int((elapsed - DT) * 2):
                    self.get_logger().info("POUR: phase=turn_left (%.1fs, wrist2=%.3f)" % (elapsed, wrist2))
            elif elapsed < self.T_POUR_HOLD + 2.0 * self.T_POUR_TURN:
                tau = min(max((elapsed - self.T_POUR_HOLD - self.T_POUR_TURN) / self.T_POUR_TURN, 0.0), 1.0)
                spline = 3.0 * tau * tau - 2.0 * tau * tau * tau
                wrist2 = self.wrist2_rest + (pi) * (1.0 - spline)  # turn back 180 right
                if int(elapsed * 2) != int((elapsed - DT) * 2):
                    self.get_logger().info("POUR: phase=turn_right (%.1fs, wrist2=%.3f)" % (elapsed, wrist2))
            else:
                wrist2 = self.wrist2_rest  # hold
                if int(elapsed) != int((elapsed - DT)):
                    self.get_logger().info("POUR: phase=hold_done (elapsed=%.1fs)" % elapsed)
                # Pour motion done -> return to waiting
                if elapsed >= self.T_POUR_HOLD + 2.0 * self.T_POUR_TURN + 0.5:
                    self.state = states.GOING_TO_WAITING
                    self.returning_from_pour = True
                    self.lift_start_q = self.q_cmd[:5].copy()
                    self.t_start_return = t
                    self.get_logger().info("Pour done. Returning to waiting pose.")
            self.q_cmd[4] = wrist2
            self.qdot_cmd = np.zeros(6)

        elif self.state == states.GOING_TO_BOTTLE:
            if self.touch_start_tip_pos is None:
                self.get_logger().info(f"Going to bottle at point: {self.p_touch}")
                p0, _, _, _ = self.chain.fkin(self.q_cmd[:5])
                self.touch_start_tip_pos = np.array(p0)
                self.t_start_bottle = t
                self.wrist1_start = float(self.q_cmd[3])
                self.wrist1_target = self.wrist1_angle(
                    self.p_touch[0] * 100.0, self.p_touch[1] * 100.0
                )

            ptip, _, Jv, _ = self.chain.fkin(self.q_cmd[:5])
            ptip = np.array(ptip)
            Jv   = np.array(Jv)

            pc, vc = self.goto(t - self.t_start_bottle, self.t_going_to_waiting,
                               self.touch_start_tip_pos,
                               self.p_touch)
            # Note: im not sure if htis should be t-self.t_start_bottle or just t

            # pc[2] = max(pc[2], self.table_clearance)
            #print(pc[2])
            ep = pc - ptip
            vr = vc + self.lam * ep

            I = np.eye(3)
            A = Jv @ Jv.T + (self.mu**2) * I
            y = np.linalg.solve(A, vr)
            qdot = Jv.T @ y

            self.q_cmd    = self.enforce_elbow_up(self.q_cmd[:5] + DT * qdot[:5])
            self.qdot_cmd = qdot[:5]
            # Spline wrist1 from start to target over same duration as tip motion
            tau = min(max((t - self.t_start_bottle) / self.t_going_to_waiting, 0.0), 1.0)
            spline = 3.0 * tau * tau - 2.0 * tau * tau * tau
            self.q_cmd[3] = self.wrist1_start + (self.wrist1_target - self.wrist1_start) * spline

            # Arrived at bottle -> close gripper, then return
            if (t - self.t_start_bottle) >= self.t_going_to_waiting:
                self.state = states.CLOSING_GRIPPER
                self.t_close_start = t
                self.get_logger().info("At bottle, closing gripper")

        elif self.state == states.CLOSING_GRIPPER:
            # Hold arm at bottle, gripper closed (handled in send below)
            self.qdot_cmd = np.zeros(6)
            if (t - self.t_close_start) >= self.T_GRIP:
                self.state = states.GOING_TO_WAITING
                self.returning_from_pour = False
                self.lift_start_q = self.q_cmd[:5].copy()
                self.t_start_return = t
                self.get_logger().info("Returning to waiting pose")

        elif self.state == states.WAITING:
            if self.pending_bottle is not None:
                self.busy = True
                self.p_touch = self.pending_bottle
                self.pending_bottle = None

                self.starttime = self.get_clock().now()
                self.touch_start_tip_pos = None
                self.t_start_bottle = 0.0

                self.state = states.GOING_TO_BOTTLE
                self.get_logger().info(f"Leaving waiting -> going to bottle at {self.p_touch}")

        
        """"
        # if contact occurs --> stop arm instantly 
        if (self.contact_hold_start_t is not None):
            hold_t = t-self.contact_hold_start_t
            self.q_cmd = self.actpos.copy()[:5]
            self.qdot_cmd = np.zeros(6)
            #tau = self.gravity(self.actpos) 
            if len(self.q_cmd) == 5:
                self.q_cmd = np.append(self.q_cmd, self.gripper_open)
            if len(self.qdot_cmd) == 5:
                self.qdot_cmd = np.append(self.qdot_cmd, 0.0)
            self.sendcmd(self.q_cmd, self.qdot_cmd)

            # if (hold_t >= self.contact_hold_t):
            #     self.contact_hold_start_t = None
            #     self.contact_return = True
            #     self.contact_return_start_t = t
            #     self.contact_return_q0 = self.actpos.copy()
            # return

        # if self.contact_return:
        #     q, qdot = self.goto(t - self.contact_return_start_t, t_contact_return, self.contact_return_q0, self.waiting)
        #     self.q_cmd    = self.enforce_elbow_up(q)
        #     self.qdot_cmd = qdot
        #     tau = self.gravity(self.actpos) 
        #     self.sendcmd(self.q_cmd, self.qdot_cmd, tau)
        #     if (t >= t_contact_return):
        #         self.contact_return = False
        #         self.contact_ignore = False
        #         self.starttime = self.get_clock().now()
        #         self.lift_start_q = None
        #         self.touch_start_tip_pos = None
        #         self.return_initialized = False
        #         if self.strip_tap_state == 2:
        #             self.strip_tap_state = 0
        #             self.busy = False
        #     return
            
        # Disk tap or strip tap (first or second side)
        # if self.strip_tap_state == 0:
        #     pass
        if self.strip_tap_state == 1:
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
                
         # Lift shoulder
        if t < t1:
            if self.lift_start_q is None:
                self.lift_start_q = self.actpos.copy()[:5]

            q0 = self.lift_start_q.copy()
            qf = q0.copy()
            qf[1] = 0.0

            # self.get_logger().info(f"length of q0: {len(q0)}, q0: {q0}, length of qf: {len(qf)}, qf: {qf}")
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
                p0, _, _, _ = self.chain.fkin(self.q_cmd[:5])
                self.touch_start_tip_pos = np.array(p0)

            ptip, _, Jv, _ = self.chain.fkin(self.q_cmd[:5])
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

            self.q_cmd    = self.enforce_elbow_up(self.q_cmd[:5] + DT * qdot[:5])
            self.qdot_cmd = qdot[:5]

        # Hold
        elif t < th:
            self.qdot_cmd = np.zeros(6)

        # Return to waiting
        elif t < t5:
            if not self.return_initialized:
                self.return_initialized = True
                self.return_q0 = self.q_cmd.copy()

            self.get_logger().info(f"length of return_q0: {len(self.return_q0)}")
            q, qdot = self.goto(t - th, self.T5,
                                self.return_q0[:5],
                                self.waiting)
            self.q_cmd    = self.enforce_elbow_up(q)
            self.qdot_cmd = qdot

        else:
            self.qdot_cmd = np.zeros(6)

        # tau = self.gravity(self.actpos) 
        # event = False
        # if not self.contact_ignore:
        #     event = self.contact_detection(tau)

        # if event and (not self.contact_ignore):
        #     self.get_logger().info("CONTACT EVENT. Stopping, holding 1s, then will return to waiting position.")
        #     self.contact_ignore = True
        #     self.contact_hold_start_t = t
        #     self.q_cmd = self.actpos.copy()
        #     self.qdot_cmd = np.zeros(3)
        """

        # Send: gripper open while waiting or going to bottle, closed at bottle / return / pour
        if len(self.q_cmd) == 5:
            if self.state in (states.WAITING, states.GOING_TO_BOTTLE):
                self.q_cmd = np.append(self.q_cmd, self.gripper_open)
            else:
                self.q_cmd = np.append(self.q_cmd, self.gripper_closed)
            self.qdot_cmd = np.append(self.qdot_cmd, 0.0)

        tau = self.gravity(self.actpos)

        # gravity compensation testing
        #qdBOB = []
        #qddotBOB = []
        #self.sendcmd(qdBOB, qddotBOB, tau)
        
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


# Robotic Bartender

The term project for **Caltech ME 134**, consisting of a custom five-joint HEBI robotic arm with a gripper. Named Chad, this robot was built from scratch and uses computer vision to locate bottles, pour into a glass, shake it, and serve it to a customer.

## Project outline

1. Fabricate the arm links and base U-bracket.
2. Model the assembled arm in URDF and calibrate its six actuators through ROS2 and HEBI feedback.
3. Detect colored bottles using HSV segmentation and map camera pixels into tabletop coordinates with ArUco markers.
4. Generate smooth Cartesian trajectories and track them with damped least-squares Jacobian inverse kinematics, gravity compensation, and wrist/gripper control.
5. Coordinate grasping and pouring through a finite-state controller.
6. Set up recovery modes for collision, a messy bartop, and a dropped bottle.

## Mechanical design

The arm links were designed in CAD and lasercut from wood. The base U-bracket and drink holder were also CAD-ed, then 3D-printed. The assembled mechanism uses five powered arm joints (base, shoulder, elbow, and two wrist axes) plus a gripper; URDF models and STL meshes are used for the geometry and kinematic chain.

## Vision and localization

The alcohol bottles used are plastic and uniform dimensions, and they contain colored beads to model the movement of liquid without actually using liquid. They each have differently colored construction paper taped to the top of the bottle, visible from an overhead camera. The `chaddetector` node converts RGB camera images to HSV and extracts bottle centers from area-filtered contours using image moments. Four ArUco markers (IDs 1–4) define a planar perspective transform between image and world coordinates. Detections are published as `geometry_msgs/Point` and `project_msgs/ObjectArray` messages, alongside diagnostic images.

## Arm control and manipulation

The controller reads `/joint_states` and uses a URDF-based kinematic chain to compute the end-effector position and translational Jacobian. At 100 Hz, damped least-squares inverse kinematics converts Cartesian tracking error into joint velocities using `Jᵀ(JJᵀ + μ²I)⁻¹(v_des + λe)`, with `μ = 0.02` and `λ = 12`. Cubic time-scaling smooths the motions, while feedforward gravity torques compensate for the shoulder, elbow, and wrist loads.

In `sixdof.py`, the state machine moves from an elbow-up waiting pose to a detected blue bottle, aligns the wrist using an empirically fitted angle, closes the gripper, moves to a fixed pouring location, and rotates the pouring wrist 180° before returning it and homing the arm. An alternative trajectory architecture (`brain_node.py` and `trajectory_node.py`) uses queued quintic joint-space splines; its current brain-node implementation only sends a waiting-pose segment.

## Recovery and fault handling

The code includes several distinct recovery mechanisms. Bottle detections can be deferred until the arm reaches its waiting pose, and new commands are ignored during active grasping/pouring. The trajectory executor supports `/trajectory/halt`, which clears queued waypoints, aborts the active spline, and publishes `/trajectory/contact`.

An earlier controller (`touchtable.py`) also implements **contact-triggered recovery**: it compares commanded and measured joint position, velocity, and effort; on a detected discrepancy, it cancels the trajectory, holds the arm for one second, and commands a two-second return to the waiting pose. This contact-detection path is **commented out in the newer `sixdof.py` pouring controller**, so it should be treated as a separate prototype rather than an active safety feature of that mode.

## Repository contents

| Folder | Contents |
| --- | --- |
| `src-tmp/chad/` | Arm controllers, state machines, kinematics, URDFs, CAD meshes, and ROS 2 launch files |
| `src-tmp/detectors/` | HSV bottle detection, ArUco coordinate mapping, and camera utilities |
| `src-tmp/hebiros/` | HEBI actuator commands, feedback, and tuning |
| `src-tmp/threedof/` | Earlier 3-DOF manipulation experiments |
| `src-tmp/demo134/` | Initial actuator and joint-motion demonstrations |
| `src-tmp/project_msgs/` | Custom object and trajectory message definitions |
| `src-tmp/chad_interfaces/` | Additional ROS 2 message and service definitions |
| `src-tmp/usb_cam/` | ROS 2 USB camera driver |

![Example Image](example.png)

![Video of Robotic Bartender]([example.png](https://tinyurl.com/bartenderbotvid))

## Authors

Rupali Batta, Claire Hays, and Ama Obeng  
California Institute of Technology, ME 134

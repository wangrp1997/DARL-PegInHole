"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import glfw
import numpy as np
import mujoco as mj
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations
from abr_control.controllers import path_planners
from robot_control import UR5Control


# initialize our robot config for the jaco2
robot_config = arm("ur5e.xml",folder='/home/rw/桌面/soft_grasp/ur5', use_sim_state=False)

# create our path planner
n_timesteps = 2000

# create our interface
dt = 0.002
interface = Mujoco(robot_config, dt=dt)
interface.connect()
interface.init_state(robot_config.START_ANGLES)
path_planner = path_planners.InverseKinematics(robot_config)
flag = True
target = None
# ur5_control = UR5Control(interface, robot_config)

try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    count = 0
    while 1:
        if glfw.window_should_close(interface.viewer.window):
            break
        if flag:
            # ur5_control.move_down(dz=0.3)
            feedback = interface.get_feedback()
            Tx = robot_config.Tx("EE", q=feedback["q"], object_type="body")
            target_xyz = np.array([Tx[0], Tx[1], Tx[2]-0.3])
            R = robot_config.R("EE", q=feedback["q"])
            target_orientation = transformations.euler_from_matrix(R, "sxyz")
            # update the position of the target
            interface.set_mocap_xyz("target", target_xyz)

            # can use 3 different methods to calculate inverse kinematics
            # see inverse_kinematics.py file for details
            path_planner.generate_path(
                                        position=feedback["q"],
                                        target_position=np.hstack([target_xyz, target_orientation]),
                                        method=1,
                                        dt=0.003,
                                        n_timesteps=n_timesteps,
                                        plot=False,)
            # returns desired [position, velocity]
            while path_planner.n < path_planner.n_timesteps:
                target = path_planner.next()[0]
                # use position control
                interface.set_target_ctrl(target[: robot_config.N_JOINTS])
                interface.viewer.render()
            flag = False

        interface.viewer.render()
        # count += 1
        # interface.send_target_angles(target[:robot_config.N_JOINTS])



finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")

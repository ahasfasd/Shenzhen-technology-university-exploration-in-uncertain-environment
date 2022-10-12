
import os
import time
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import pybullet as p
from ur5_envs import models
from ur5_envs.grippers import Suction
from utils import pybullet_utils, utils
import ur5_envs.pb_ompl as pb_ompl

# PLACE_STEP = 0.0003
# PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = "racecar/racecar.urdf"

# UR5_URDF_PATH = "ur5/ur5.urdf"
class UR5:
    '''
    UR5 Robot
    
    Args:
        - `ee_type`: str
            * end effector type, we only have 'suction' currently
        - `speed`: float
        - `obj_ids`: dict
            * collect all the graspable object we have in the scene, for Suction to grasp object
        -  `homej`: list(float) [3]
            * init joint positions
        - `is_static`: bool
            * True if the robot is moving

    Functions:
        - `reset` -> reload the robot state to init pose
        - `add_object_to_list` -> add obj id into self.obj_ids
        - `movej` -> move end effector to target joints' positions
        - `movep` -> move end effector to target pose
        - `solve_ik` -> solve ik for target pose
        - `move_to` -> controll end effector to approach target pose
        - `pick` -> move to target pose to pick object
        - `place` -> move to target pose to place object
        - `get_ee_pose` -> get ee pose
        - `debug_gui` -> add slide bar gui in the pybullet gui
        - `update_arm` -> update the joints' positions based on slide bar values
    '''
    
    def __init__(self, ee_type='suction', speed=0.01):
        

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # init joint posiitons
        self.homej = np.array([0, 0, 0, 0, 0, 0])

        self.speed = speed

        self.ee_type = ee_type

    def reset(self, show_gui=True):
        """Performs common reset functionality for all supported tasks."""

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = pybullet_utils.load_urdf(
            p, os.path.join( models.get_data_path(), UR5_URDF_PATH), [0, -2.3, 0], [0, 0, 1, 1] )

        self.ee_tip = 10
        if self.ee_type == 'suction':
            self.ee = Suction(models.get_data_path(), self.ur5, 14, self.obj_ids)
            self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.joint_names = [str(j[1]) for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.joint_lower_limits = [j[8] for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.joint_upper_limits = [j[9] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        self.set_joints_state(self.homej)
        # self.movej(self.homej)

        # Reset end effector.
        # self.ee.release()

        # add ompl
        self.robot_ompl = pb_ompl.PbOMPLRobot(self.ur5, self.homej)

        if show_gui:
            self.debug_gui()


    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
            for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object_to_list(self, category, obj_id):
        self.obj_ids[category].append(obj_id)

    #---------------------------------------------------------------------------
    # Robot Movement Functions
    #---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
            # time.sleep(1/ 24.0)
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints


    # TODO check reachable

    #============----------------   Move and grasp   ----------------============#
    
    def move_to(self, pose, offset=(0,0,0)):
        """Move end effector to pose.

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        # 先移动到 offset 位置，再不断靠近目标
        offset = np.array(offset)
        
        prepick_to_pick = (offset, (0,0,0,1))

        prepick_pose = utils.multiply(pose, prepick_to_pick)
        pybullet_utils.draw_pose(prepick_pose, life_time=2)
        timeout = self.movep(prepick_pose)

        print("ee position is: ", p.getLinkState(2, 10))
        if np.linalg.norm(offset) == 0:
            return timeout
        
        # Move towards pick pose until contact is detected.
        delta_step = 103
        delta = (-offset / 100.0 , (0,0,0,1))

        targ_pose = prepick_pose
        # 改成 and 判断的话就不会一直往下怼
        # 改成 or 判断的话就一直往下怼
        while delta_step >= 0 and not self.ee.detect_contact():  # and target_pose[2] > 0:
            delta_step -= 1
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= self.movep(targ_pose)
            if timeout:
                return True

        return timeout

    def pick(self, pose: object) -> object:
        """Move to pose to pick.

        Args:
            pose: SE(3) picking pose.

        Returns:
            pick_success: pick success if True
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        
        if timeout:
            return False

        # Activate end effector, move up, and check picking success.
        self.ee.activate()
        
        postpick_to_pick = ( offset, (0,0,0,1) )
        
        postpick_pose = utils.multiply(pose, postpick_to_pick)

        timeout |= self.movep(postpick_pose, self.speed)

        pick_success = self.ee.check_grasp()

        return pick_success

    def place(self, pose):
        """Move end effector to pose to place.

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        if timeout:
            return True
        
        self.ee.release()

        postplace_to_place = (offset, (0,0,0,1.0))
        postplace_pose = utils.multiply(pose, postplace_to_place)
        timeout |= self.movep(postplace_pose)

        return timeout

    #============----------------   robot tools   ----------------============#
    
    def get_ee_pose(self):
        pos, orient, *_ = p.getLinkState(self.ur5, self.ee_tip)
        return np.array(pos), np.array(orient)
        
    def debug_gui(self):
        debug_items = []
        joints = self.get_current_joints()
        print(joints)
        for i in range(len(joints)):
            item = p.addUserDebugParameter(self.joint_names[i], self.joint_lower_limits[i], self.joint_upper_limits[i], joints[i])
            print(i, p.readUserDebugParameter(item))
            debug_items.append(item)
        self.debug_items = debug_items

    def update_arm(self):
        joint_values = []
        for item in self.debug_items:
            joint_values.append(p.readUserDebugParameter(item))
        self.movej(joint_values)

    def get_current_joints(self):
        return np.array([p.getJointState(self.ur5, i)[0] for i in self.joints])

    def set_joints_state(self, joints):
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], joints[i])

    def get_ur5_pose(self):
        pos, ori = p.getBasePositionAndOrientation(self.ur5)
        return np.array(pos), np.array(ori)
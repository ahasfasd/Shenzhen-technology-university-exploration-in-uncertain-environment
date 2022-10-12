from ur5_envs.envs import Env
import pybullet as p
from utils import pybullet_utils
from utils import utils
import os
import ur5_envs.pb_ompl as pb_ompl
import time
import common_param
import copy
import math
import numpy as np
import DRL.DDPG as parm
from PIL import Image
import mediapipe as mp

UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


class gym_ur5(Env):

    def __init__(self,
                 assets_root,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 use_egl=False) -> None:
        super().__init__(assets_root, disp=disp, shared_memory=shared_memory, hz=hz, use_egl=use_egl)

        self.observation = None
        self.target = None
        self.target_radius = None
        self.init_obstacle = False
        self.start()

        # self.reset()

    def init_obstalces(self, num_box, use_static):
        # add table
        # self.init_obstacle = True
        # self.add_obstalce([0.001, 0.001, -1], [0, 0, 0, 1], [0.05, 0.05, 0.05])
        # add obstalces
        if use_static and num_box == 3:
            pos = [[3, 3, 1], [6, 4, 0], [6, 3, 0]]
            self.obstacle_size = [[0.04, 0.04, 0.25], [0.02, 0.03, 0.3], [0.03, 0.05, 0.2]]
            for i in range(len(pos)):
                self.add_obstalce(pos[i], [0, 0, 0, 1], self.obstacle_size[i], utils.COLORS_A['blue'])
        elif use_static and num_box == 1:
            pos = [[np.random.random() * 2 - 1, np.random.random() * 2 - 1, 0]]
            pos1 = [[np.random.random() * 2 - 1, np.random.random() * 2 - 1, 0]]
            pos2 = [[np.random.random() * 2 - 1, np.random.random() * 2 - 1, 0]]
            self.obstacle_size = [[0.05, 0.05, 0.05]]
            for i in range(len(pos)):
                self.add_obstalce(pos[i], [0, 0, 0, 1], self.obstacle_size[i], utils.COLORS_A['blue'])
            for i in range(len(pos1)):
                self.add_obstalce(pos1[i], [0, 0, 0, 1], self.obstacle_size[i], utils.COLORS_A['blue'])
            for i in range(len(pos2)):
                self.add_obstalce(pos2[i], [0, 0, 0, 1], self.obstacle_size[i], utils.COLORS_A['blue'])
        else:
            pass
            # pos = np.random.rand(num_box, 3)
            # size = np.random.rand(num_box, 3)*0.1 + 0.05
            # for i in range(num_box):

        # add dummy
        dummy_pos_x = np.random.random() * 2 - 1
        dummy_pos_y = np.random.random() * 2 - 1
        dummy_pos_z = 0
        # self.target_radius = 0.1
        self.target = self.add_dummy([dummy_pos_x, dummy_pos_y, dummy_pos_z], self.target_radius)
        # else:
        #     self.move_object(self.target, [dummy_pos_x, dummy_pos_y, dummy_pos_z], [0, 0, 0, 1])
        #     # print(dummy_pos_x, dummy_pos_y, dummy_pos_z)
        #     # print("target poisition: ", self.get_object_pose(self.target))

    def get_target(self):
        pos, ori = self.get_object_pose(self.target)
        return np.array(pos), np.array(ori)

    def get_ee_target_distance(self, mode):
        x1 = self.robot.get_ee_pose()
        x2 = self.get_target()
        if mode == common_param.CARTESIA_DISTANCE:
            return np.sqrt(np.sum(np.square(x2[0] - x1[0])))
        elif mode == common_param.MIX_DISTANCE:
            pass

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration in env."""
        t0 = time.time()
        TimeOut = False
        currj = [p.getJointState(self.robot.ur5, i)[0] for i in self.robot.joints]
        currj = np.array(currj)
        diffj = targj - currj
        diffj /= 100
        # Move with constant velocity
        # norm = np.linalg.norm(diffj)
        # v = diffj / norm if norm > 0 else 0
        # stepj = currj + v * speed
        stepj = currj
        gains = np.ones(len(self.robot.joints))
        for i in range(100):
            stepj += diffj
            collision = self.check_collision()
            # if collision:
            #     break
            p.setJointMotorControlArray(
                bodyIndex=self.robot.ur5,
                jointIndices=self.robot.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
        # diffj = targj - currj
        # if all(np.abs(diffj) < 1e-2):
        #     return True, TimeOut
        #
        # if self.check_collision():
        #     return False, TimeOut
        #
        #
        # # time.sleep(1/ 24.0)
        # TimeOut = True
        # print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        # return False, TimeOut

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose in env."""
        targj = self.robot.ur5.solve_ik(pose)
        return self.movej(targj, speed)

    def joints_step(self, joints):
        curj = self.robot.get_current_joints()

        curj += joints

        self.movej(curj)
        # self.robot.set_joints_state(curj)

        # 深拷贝环境信息并加入机械臂关节值
        observation = copy.deepcopy(self.observation)
        # observation = np.append(observation, self.robot.get_current_joints())
        pos ,ori = self.robot.get_ur5_pose()
        observation = np.append(observation, pos)
        observation = np.append(observation, ori)
        observation = np.append(observation, self.robot.get_current_joints())
        # print('123456 ',observation)
        # print(self.robot.get_current_joints())

        # 计算欧氏距离当做reward
        reward = self.get_ee_target_distance(common_param.CARTESIA_DISTANCE)
        done = False
        # 当小于目标半径时结束

        # contactPoints = p.getContactPoints(self.robot.ur5, self.target)
        # if len(contactPoints) > 0:
        #     # print(2, contactPoints)
        #     done = True

        if reward < 0.2:
            done = True

        info = []
        # info = {"planeSuccess": planSuccess,
        #         "timeOut": timeOut}
        return observation, reward, done, info

    def pos_step(self, pos):
        curp = self.robot.get_ee_pose()[0]
        curp += pos

        self.movep(curp)


    def reset(self, show_gui=True):
        '''load plane and robot'''
        while len(self.boxes) > 0:
            id1 = self.boxes.pop()
            id2 = self.obstacles.pop()
            if id1 == id2:
                self.remove_obstalce(id1)
            else:
                print("something error!!!")
                break

        while len(self.dummy) > 0:
            id = self.dummy.pop()
            self.remove_obstalce(id)

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # reload init scene
        plane = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                         [0, 0, -0.001])
        # work_space = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
        # [0.5, 0, 0])

        self.obstacles.append(plane)
        # self.obstacles.append(work_space)
        self.init_obstalces(parm.OBSTACLE_NUM, True)

        # reset robot / reload robot
        self.robot.reset(show_gui)
        # self.base_box = self.add_obstalce([3, 4, 0], [0, 0, 0, 1], [0.15, 0.15, 0.15])
        # self.move_object(self.robot.ur5, [0, 0, 0.3], [0, 0, 0, 1])

        # pos_, _ = p.getBasePositionAndOrientation(self.target)
        # froms = [[pos_[0] - 0.11, pos_[1] + 0.11, 0], [pos_[0] - 0.11, pos_[1] - 0.11, 0],
        #          [pos_[0] + 0.11, pos_[1] - 0.11, 0], [pos_[0] + 0.11, pos_[1] + 0.11, 0],
        #          [pos_[0] - 0.11, pos_[1] + 0.11, 0.2], [pos_[0] - 0.11, pos_[1] - 0.11, 0.2],
        #          [pos_[0] + 0.11, pos_[1] - 0.11, 0.2], [pos_[0] + 0.1, pos_[1] + 0.11, 0.2]]
        # tos = [[pos_[0] - 0.11, pos_[1] - 0.11, 0], [pos_[0] + 0.11, pos_[1] - 0.11, 0],
        #        [pos_[0] + 0.11, pos_[1] + 0.11, 0], [pos_[0] - 0.11, pos_[1] + 0.11, 0],
        #        [pos_[0] - 0.11, pos_[1] - 0.11, 0.2], [pos_[0] + 0.11, pos_[1] - 0.11, 0.2],
        #        [pos_[0] + 0.11, pos_[1] + 0.11, 0.2], [pos_[0] - 0.11, pos_[1] + 0.11, 0.2]]
        # for i in  froms:
        #     for j in tos:
        #         p.addUserDebugLine(
        #             lineFromXYZ=i,
        #             lineToXYZ=j,
        #             lineColorRGB=[0, 1, 0],
        #             lineWidth=2
        #         )


        # TODO reset task

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot.robot_ompl, self.obstacles)

        self.pb_ompl_interface.set_planner("BITstar")

        # self.observation = np.array([])
        # 加入障碍物位姿
        # for i in range(len(self.boxes)):
        #     pos, ori = self.get_object_pose(self.boxes[i])
        #     self.observation = np.append(self.observation, pos)
        #     self.observation = np.append(self.observation, ori)
        #     self.observation = np.append(self.observation, self.boxes_sizes[i])

        # 加入目标点
        # pos, ori = self.get_target()
        # self.observation = np.append(self.observation, pos)
        # self.observation = np.append(self.observation, ori)

        # observation = copy.deepcopy(self.observation)
        observation = []
        pos , ori = self.robot.get_ur5_pose()
        observation = np.append(observation, pos)
        observation = np.append(observation, ori)



        box_3d_points = []
        p.stepSimulation()

        viewMatrix = p.computeViewMatrix([2, 0, 2],
                                         [0, 0, 0],
                                         [0, 0, 1]
                                         )
        # print(viewMatrix)
        projectionMatrix = p.computeProjectionMatrixFOV(55, 1, 0.01, 10, 0)

        _w, _h, RGB_Pixels, deph_Pixels, seg_Pixels = p.getCameraImage(2048, 2048, viewMatrix=viewMatrix,
                                                                       projectionMatrix=projectionMatrix)

        # print(RGB_Pixels.shape)
        # print(deph_Pixels.shape)
        # print(seg_Pixels.shape)
        # image.convert("RGB")

        image = Image.fromarray(RGB_Pixels)
        # image.show()
        # a=1
        image = image.convert("RGB")
        # image.show()
        image = np.array(image)

        obs_flag = True


        with mp_objectron.Objectron(static_image_mode=True,
                                    max_num_objects=1,
                                    min_detection_confidence=0.1,
                                    model_name='Cup') as objectron:

            results = objectron.process(image)
            # Draw box landmarks.
            if not results.detected_objects:
                # print('not Cup')
                obs_flag = False
                for zero in range(27):
                    box_3d_points.append(0)
            else:
                # print('Cup')
                for Landmark in results.detected_objects[0].landmarks_3d.landmark:
                    box_3d_points.append(Landmark.x)
                    box_3d_points.append(Landmark.y)
                    box_3d_points.append(Landmark.z)
                # annotated_image = image.copy()
                # for detected_object in results.detected_objects:
                #     mp_drawing.draw_landmarks(
                #         annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)  # 画框
                #     mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                #                          detected_object.translation)  # 画轴
                    # cv2.imwrite('C:\\Users\\BOB\\Downloads\\' + str(idx) + '.png', annotated_image)
                    # cv2.imshow('1',annotated_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

        with mp_objectron.Objectron(static_image_mode=True,
                                    max_num_objects=3,
                                    min_detection_confidence=0.1,
                                    model_name='Chair') as objectron:

            results = objectron.process(image)
            # Draw box landmarks.
            if not results.detected_objects or len(results.detected_objects)<3:
                # print('Not Chair')
                obs_flag = False
                for zero in range(27):
                    box_3d_points.append(0)
            else:
                # print('Chair:')
                for j in range(len(results.detected_objects)):
                    for Landmark in results.detected_objects[j].landmarks_3d.landmark:
                        box_3d_points.append(Landmark.x)
                        box_3d_points.append(Landmark.y)
                        box_3d_points.append(Landmark.z)
                # annotated_image = image.copy()
                # for detected_object in results.detected_objects:
                #     mp_drawing.draw_landmarks(
                #         annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)  # 画框
                #     mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                #                          detected_object.translation)  # 画轴
                    # cv2.imwrite('C:\\Users\\BOB\\Downloads\\' + str(idx) + '.png', annotated_image)
                    # cv2.imshow('1',annotated_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        # self.
        box_3d_points = np.array(box_3d_points)
        # print(box_3d_points.shape)
        # gym_ur5.alter_observation(box_3d_points)
        self.observation = box_3d_points
        observation = np.append(self.observation, observation)

        observation = np.append(observation, self.robot.get_current_joints())


        return observation, obs_flag

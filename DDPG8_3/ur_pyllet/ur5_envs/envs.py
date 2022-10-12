import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm
import pybullet as p

import os
import pkgutil
import sys
import tempfile
import time

import numpy as np
from ur5_envs import models
from ur5_envs import cameras

from utils import pybullet_utils
from utils import utils
from utils import pack
from ur5_envs.ur5 import UR5
import ur5_envs.pb_ompl as pb_ompl

UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"
CHAIR_OBJ_PATH = "chair/chair.obj"

CUP_OBJ_PATH = "cup/mug.obj"

colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']


class Env:
    def __init__(self,
                 assets_root,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 use_egl=False) -> None:
        if use_egl and disp:
            raise ValueError('EGL rendering cannot be used with `disp=True`.')

        self.use_egl = use_egl
        self.disp = disp
        self.hz = hz
        self.shared_memory = shared_memory
        self.assets_root = assets_root
        self.obstacles = []
        self.boxes = []
        self.boxes_sizes = []
        self.dummy = []

        # self.pix_size = 0.003125

        self.seed(666)
        # add cameras
        self.cams = {}
        self.cams['middle'] = cameras.Camera(cameras.RealSenseD415.CONFIG[0], self._random)
        self.cams['left'] = cameras.Camera(cameras.RealSenseD415.CONFIG[1], self._random)
        self.cams['right'] = cameras.Camera(cameras.RealSenseD415.CONFIG[2], self._random)

        # add robot
        self.robot = UR5()
        print("init homej is: ", self.robot.homej)
        # TODO add task

    def start(self, show_gui=True):
        '''Start PyBullet'''

        disp_option = p.DIRECT
        if self.disp:
            disp_option = p.GUI
        if self.shared_memory:
            disp_option = p.SHARED_MEMORY

        client = p.connect(disp_option)

        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=self.assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        self._egl_plugin = None
        if self.use_egl:
            assert sys.platform == 'linux', ('EGL rendering is only supported on '
                                             'Linux.')
            egl = pkgutil.get_loader('eglRenderer')
            if egl:
                self._egl_plugin = p.loadPlugin(egl.get_filename(),
                                                '_eglRendererPlugin')
            else:
                self._egl_plugin = p.loadPlugin('eglRendererPlugin')
            print('EGL renderering enabled.')

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, show_gui)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(self.assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / self.hz)
        p.setGravity(0, 0, -9.8)

        # If using --disp, move default camera closer to the scene.
        if self.disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

    def reset(self, show_gui=True):
        '''load plane and robot'''

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # reload init scene
        plane = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                         [0, 0, -0.001])
        work_space = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
                                              [0.5, 0, 0])

        self.obstacles.append(plane)
        self.obstacles.append(work_space)

        # TODO reset task

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # reset robot / reload robot
        self.robot.reset(show_gui)

        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot.robot_ompl, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

    def take_images(self, cam_name: str):
        rgb, dep, seg = self.cams[cam_name].take_images()
        return rgb, dep, seg

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0

        obj_id = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base
        )
        self.robot.add_object_to_list(category, obj_id)
        self.obstacles.append(obj_id)
        self.boxes.append(obj_id)
        return obj_id

    def add_box(self, pose, size, color=None, category='rigid'):

        box_template = 'box/box-template.urdf'

        urdf = pybullet_utils.fill_template(self.assets_root, box_template, {'DIM': size})
        box_id = self.add_object(urdf, pose, category)
        os.remove(urdf)

        if color is None:
            color = pybullet_utils.color_random(utils.COLORS['brown'])

        p.changeVisualShape(box_id, -1, rgbaColor=color)

        p.changeDynamics(box_id, -1, mass=0.5)

        return box_id

    def add_obstalce(self, box_pos, box_ori, half_box_size, color=utils.COLORS_A['white']):
        # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        scale = [0.007, 0.007, 0.007]
        colBoxId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName= os.path.join( models.get_data_path(),CHAIR_OBJ_PATH) ,

            meshScale=scale)

        # visual_id = p.createVisualShape(shapeType=p.GEOM_BOX,
        #                                 fileName=os.path.join( models.get_data_path(), CHAIR_STL_PATH),
        #                                 rgbaColor=color,
        #                                 visualFramePosition = box_pos
        #                                 )
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=os.path.join( models.get_data_path(),CHAIR_OBJ_PATH),
            rgbaColor=[0, 0, 1, 1],

            visualFramePosition=[0, 0, 0],
            meshScale=scale)

        # box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos,
        #                            baseOrientation=box_ori, baseVisualShapeIndex=visual_id)
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=colBoxId,
            baseVisualShapeIndex=visual_id,
            basePosition=box_pos,
            baseOrientation=[1, 0.4, 0.4, 1],)


        # box_id = pybullet_utils.load_urdf(p, os.path.join( models.get_data_path(), CHAIR_URDF_PATH),box_pos )

        # p.changeVisualShape(box_id, -1, rgbaColor=color)
        # p.changeDynamics(box_id, -1, mass=0.5)

        self.obstacles.append(box_id)
        self.boxes.append(box_id)
        # self.boxes_sizes.append(half_box_size)
        return box_id

    def add_dummy(self, sphere_pos, sphere_radius=0, color=utils.COLORS_A['green']):
        # colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0)
        # visualSphereId = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=color)
        scale = [4, 4, 4]


        visualSphereId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=os.path.join( models.get_data_path(),CUP_OBJ_PATH),
            rgbaColor=[1, 0, 1, 1],
            visualFramePosition=[0, 0, 0],
            meshScale=scale)

        # sphereId = p.createMultiBody(baseMass=0,
        #                              # baseCollisionShapeIndex = colSphereId,
        #                              baseVisualShapeIndex=visualSphereId,
        #                              basePosition=sphere_pos,
        #                              baseOrientation=[0, 0, 0, 1])

        sphereId = p.createMultiBody(
            baseMass=0,
            # baseCollisionShapeIndex=colBoxId,
            baseVisualShapeIndex=visualSphereId,
            basePosition=sphere_pos,
            # baseOrientation=[1, 0, 0, 1],
        )


        self.dummy.append(sphereId)
        return sphereId

    def move_object(self, id, pos, ori):
        return p.resetBasePositionAndOrientation(id, pos, ori)

    def check_collision(self):

        # 自碰撞检测
        # for i in range(16):
        #     j = i+2
        #     while j < 16:
        #         contactPoints = p.getContactPoints(self.robot.ur5, self.robot.ur5, i, j)
        #         if len(contactPoints) > 0:
        #             print(i,j)
        #             print(contactPoints)
        #         j += 1

        # 和障碍物进行碰撞检测
        lenth = len(self.boxes)
        # print(lenth)
        for i in range(lenth):
            contactPoints = p.getContactPoints(self.boxes[i], self.robot.ur5)
            if len(contactPoints) > 0:
                # print(1, contactPoints)
                return True

        # 和底座进行碰撞检测
        # contactPoints = p.getContactPoints(self.robot.ur5, self.boxes[lenth])
        # if len(contactPoints) > 0:
        #     # print(2, contactPoints)
        #     return True
        #
        # contactPoints = p.getContactPoints(self.robot.ur5, self.boxes[lenth])
        # if len(contactPoints) > 1:
        #     # print(2, contactPoints)
        #     return True
        # P_min, P_max = p.getAABB(self.robot.ur5)
        # # print(self.robot.ur5)
        # # print(P_min ,"    ",  P_max )
        # id_tuple = p.getOverlappingObjects(P_min, P_max)
        # if len(id_tuple) > 2:
        #     print(len(id_tuple))
        #     return True

        # # 和地板进行碰撞检测
        # contactPoints = p.getContactPoints(self.robot.ur5, self.obstacles[0])
        # if len(contactPoints) > 1:
        #     # print(3, contactPoints)
        #     return True
        return False

    def get_object_pose(self, id):
        pos, ori = p.getBasePositionAndOrientation(id)
        return np.array(pos), np.array(ori)

    def remove_obstalce(self, box_id):
        return p.removeBody(box_id)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        p.disconnect()

    def key_event(self):

        def check_key(key, keys):
            key = ord(key)
            return key in keys and keys[key] & p.KEY_WAS_TRIGGERED

        keys = p.getKeyboardEvents()

        if check_key('a', keys):
            unit = 0.025
            pos = np.random.rand(3) * 0.3 + [0, 0, 0.1]
            quat = utils.eulerXYZ_to_quatXYZW(np.random.rand(3) * np.math.pi)
            size = np.random.randint(1, 6, 3) * unit
            self.add_box([pos, quat], size)

        elif check_key('b', keys):
            pos = np.random.rand(3) * 0.3 + [0, 0, 0.1]
            quat = utils.eulerXYZ_to_quatXYZW(np.random.rand(3) * np.math.pi)
            obj_id = self.add_object('bottle_21/bottle_21.urdf', [pos, quat], category='fixed')
            p.changeDynamics(obj_id, -1, mass=0.5)

        elif check_key('r', keys):
            # self.reset()
            # self.robot.ee.release()
            if len(self.obstacles) > 0:
                id = self.obstacles.pop()
                self.remove_obstalce(id)
        elif check_key('t', keys):
            self.check_collision()

        else:
            pybullet_utils.key_event(keys)

    def ompl(self, goal):
        currj = self.robot.get_current_joints()
        self.pb_ompl_interface.set_obstacles(self.obstacles)

        self.robot.robot_ompl.set_state(currj)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            while True:
                self.pb_ompl_interface.execute(path)
                # for joint in path:
                #     self.env.robot.movej(joint)
        return res, path

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

import time

import numpy as np
from ur5_envs.envs import Env, colors
from ur5_envs import cameras

from utils import pybullet_utils
from utils import utils
from utils import pack
import ur5_envs.pb_ompl as pb_ompl

class TAP_Env(Env):
    def __init__(self, assets_root, disp=False, shared_memory=False, hz=240, use_egl=False) -> None:
        super().__init__(assets_root, disp=disp, shared_memory=shared_memory, hz=hz, use_egl=use_egl)

        image_size = (480, 640)
        intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

        init_position = (0.20327461, 0.05055285, 0.40699884)
        init_rotation = (np.pi / 4, 4 * np.pi / 4, 1 * np.pi / 4)
        init_rotation = p.getQuaternionFromEuler(init_rotation)

        init_config = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': init_position,
            'rotation': init_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }

        self.cams['init'] = cameras.Camera(init_config, self._random)
        # TODO
        # self.cams['tmp'] = cameras.Camera(cameras.RealSenseD415.CONFIG[3], self._random)

        # self.start(disp)
        # self.reset(disp)
        # # TODO add task
        #
        # # add OMPL
        # self.robot_ompl = pb_ompl.PbOMPLRobot(self.robot.ur5)
        # # setup pb_ompl
        # self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot_ompl, self.obstacles)
        # self.pb_ompl_interface.set_planner("BITstar")


    def init_scene(self, \
        block_tasks, \
        init_container_origin=[0.4, -0.35, 0], init_container_size=[7,7,20],\
        init_space_width=0.5, init_space_height=0.6):

        wall1 = self.add_box(((init_space_width, 0, 0), (0, 0, 0, 1)),  (init_space_width, 0.01, init_space_height), color=(0,0,0, 1), category='fixed' )
        wall2 = self.add_box(((0.25, -0.25, 0), (0, 0, 0, 1)),  (0.01, init_space_width, init_space_height), color=(0,0,0, 1), category='fixed' )
        wall3 = self.add_box(((0.25 + init_space_width, -0.25, 0), (0, 0, 0, 1)),  (0.01, init_space_width, init_space_height), color=(0,0,0, 1), category='fixed' )
        wall4 = self.add_box(((init_space_width, -init_space_width, 0), (0, 0, 0, 1)),  (init_space_width, 0.01, init_space_height), color=(0,0,0, 1), category='fixed' )

        for task in block_tasks:
            block_unit = task['unit']
            block_num = task['num']
            block_min_size = task['min_size']
            block_max_size = task['max_size']
            is_mess = task['is_mess']
            self.generate_boxes( block_num, block_min_size, block_max_size, init_container_size, init_container_origin, block_unit, is_mess )
            pybullet_utils.simulate_step(100)

        p.removeBody(wall1)
        p.removeBody(wall2)
        p.removeBody(wall3)
        p.removeBody(wall4)
        pybullet_utils.simulate_step(100)


    def add_boxes(self, zone_pose):
        # Add stack of boxes on pallet.
        margin = 0.01
        object_ids = []

        stack_size = (0.19, 0.19, 0.19)

        stack_dim = np.int32([2, 3, 3])
        # stack_dim = np.random.randint(low=2, high=4, size=3)
        box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
        for z in range(stack_dim[2]):

            # Transpose every layer.
            stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
            box_size[0], box_size[1] = box_size[1], box_size[0]

            for y in range(stack_dim[1]):
                for x in range(stack_dim[0]):
                    position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                    position[0] += x * margin - stack_size[0] / 2
                    position[1] += y * margin - stack_size[1] / 2
                    position[2] += z * margin + 0.03
                    pose = (position, (0, 0, 0, 1))
                    pose = utils.multiply(zone_pose, pose)
                    
                    box_id = self.add_box(pose, box_size)

                    object_ids.append((box_id, (0, None)))
                    pybullet_utils.color_random_brown(box_id)

    def generate_boxes(self, blocks_num, block_min_size, block_max_size, \
        container_size, container_origin, block_unit, is_mess=False, loop_num=1):
        ''' Generate blocks in simulator

        Args:
            - `blocks_num`: int
            - `block_min_size`: int
            - `block_max_size`: int
            - `container_size`: list(int) [3]
            - `container_origin`: list(float) [3]
                * left down corner of container
            - `block_unit`: float
                * how many meters each unit is
            - `is_mess`: boolean
                * True if we want to generate random placement
            - `loop_num`: int
                * how many time you want to generate blocks data (usually 1)
        
        Returns:
            - `positions`: np.array [n, 3]
            - `blocks`: np.array [n, 3]
        '''
        
        # generate blocks
        for _ in range(loop_num):
            while True:
                # blocks = np.random.randint( block_min_size, block_max_size+1, (blocks_num, 3) )
                
                size_list = [ i for i in range(block_min_size, block_max_size+1) ]
                # Gaussian distribution
                mu = 0.5
                sigma = 0.16
                prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
                prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
                prob_blocks = prob_blocks / np.sum(prob_blocks)
                blocks = np.random.choice(size_list, (blocks_num, 3), p=prob_blocks )
                
                positions, container, stable, ratio, scores = pack.calc_positions_lb_greedy(blocks, container_size)
                if np.sum(stable) == blocks_num:
                    break

        # add block into scene
        for i in range(blocks_num):
            quat = (0, 0, 0, 1)
            
            if is_mess:
                quat = utils.eulerXYZ_to_quatXYZW( np.random.rand(3) * np.math.pi / 6 )
            
            size = blocks[i] * block_unit 
            
            pos = positions[i]
            # pos[0] = container_size[0] - pos[0]
            # pos[1] = container_size[1] - pos[1]
            pos = pos * block_unit + size/2
            pos += container_origin
            # 加点高度掉落
            pos += [0, 0, 0.15]

            # 这里 size 减去一点点大小是为了生成间距的数据，这样可能避免碰撞，方便抓取规划？
            margin = 0.0002
            color = colors[np.random.randint(len(colors))]
            color = pybullet_utils.color_random(color)
            self.add_box( (pos, quat), size - margin, color )

        return positions

    def ompl_demo(self):
        self.start()
        self.reset()
        self.add_obstalce([0.7, 0, 0.4], [0, 0, 0, 1], [0.5, 0.5, 0.05])
        goal = [-3.14, 0, 0, -1.57, -3.14, 0]

        return self.ompl(goal)

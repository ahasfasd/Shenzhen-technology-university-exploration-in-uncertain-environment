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
from ur5_envs.task import Env, TAP_Env
from ur5_envs import models
from ur5_envs.gym_ur5 import gym_ur5

from rich.traceback import install

from utils import pybullet_utils
from utils import utils

install(show_locals=True)

if __name__ == "__main__":
    # env = TAP_Env(models.get_data_path(), True )
    env = gym_ur5(models.get_data_path(), disp=True)

    show_gui = True
    env.reset(show_gui)
    
    # zone_size, zone_pose = env.init_scene()
    # env.add_boxes(zone_pose)
    #
    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )
    # pybullet_utils.simulate_step(100)
    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )
    # pybullet_utils.simulate_step(100)
    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )
    #
    # pos, quat = env.robot.get_ee_pose()
    # pos[1] -= 0.1
    # pos[2] += 0.1
    # pos_b = [pos, quat]
    #
    # pos_a = [[0.64, 0, 0.15/2], [0,0,0,1]]
    # env.add_box( pos_a, [0.05, 0.1, 0.15], utils.COLORS['red'] + [1] )
    #
    # pick_pose = utils.multiply(pos_a, [[0., 0, 0.15/2], [0,0,0,1]])
    #
    # pybullet_utils.draw_pose(pos_a)
    # pybullet_utils.draw_pose(pick_pose)
    # pybullet_utils.draw_pose(pos_b)
    #
    # env.robot.pick(pick_pose)
    # env.robot.place(pos_b)
    #
    # count = 1
    # sign = 1
    #
    # rot_quat = utils.eulerXYZ_to_quatXYZW([ sign * np.math.pi / 18.0, 0, 0 ])
    # pos_c = utils.multiply(pos_b, [[0,0,0], rot_quat])
    #
    # joints = env.robot.solve_ik( ((100, 0, 0), (0, 0, 0, 1)) )
    # print(joints)

    try:
        while True:
            p.stepSimulation()
            # time.sleep(1/240.0)
            env.key_event()
            # env.robot.movej(env.robot.homej)
            # print(env.robot.get_current_joints())
            # print("homej is: ", env.robot.homej)

            # if env.robot.is_static:
            #     rot_quat = utils.eulerXYZ_to_quatXYZW([ sign * np.math.pi / 18.0, 0, 0 ])
            #     count += 1
            #     if count == 36:
            #         sign = -sign
            #         count = 1
            #     pos_c = utils.multiply(pos_c, [[0,0,0.0], rot_quat])
            #     env.robot.move_to(pos_c, (0, 0, 0.05))

            # if show_gui and env.robot.is_static:
            #      env.robot.update_arm()
            #      rgb, dep, seg = env.take_images('middle')

    except Exception as e:
        # env.close()
        print(e)
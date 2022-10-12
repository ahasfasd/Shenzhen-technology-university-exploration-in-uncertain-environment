#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
#
# sys.path.append("/home/wang/ur5_pybullet_env/ur5_pybullet_env")
# print(sys.path)
import DRL.DDPG as parm
import numpy as np
import datetime
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import pybullet as p
from ur5_envs.gym_ur5 import gym_ur5
from ur5_envs import models
from DRL.DDPG import DDPG
from DRL.OUnoise import OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt

MAX_EPISODES = 8000
MAX_EP_STEPS = 500
MEMORY_CAPACITY = 100000
c1 = 1000
c2 = 0
c3 = 20
c4 = 100


def plot(frame_idx, rewards, acc):
    plt.subplot(121)
    plt.plot(rewards)
    plt.ylabel("Total_reward")
    plt.xlabel("Episode")

    plt.subplot(122)
    plt.plot(acc)
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Episode")
    plt.show()


def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5 * dis * dis
    else:
        return delta * (dis - 0.5 * delta)


if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    # Tesnorboard
    writer = SummaryWriter(
        'runs/{}_DDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "ur5_pybullet"))
    env = gym_ur5(models.get_data_path(), disp=True)

    show_debug = False
    rl = DDPG()
    # rl.load_mode()
    var = 0.3
    total_rewards = []
    step_sums = []
    acc_epi = []
    acc = 0
    last_max = -1
    action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(parm.ENV_PARAMS['action']), 0.05)
    # 主循环

    for i in range(MAX_EPISODES):

        obs, obs_flag = env.reset(show_debug)
        while not obs_flag:
            obs, obs_flag = env.reset(show_debug)

        # action_noise.reset()
        # obs = env.fix_target(box)
        state = obs.copy()
        st = 0
        rw = 0
        success = False

        while True:

            p.stepSimulation()
            state = np.array(state)
            # print(state.shape)
            # print(state)

            # noise = np.random.normal(loc=0, scale=var, size=6)
            # print(state)
            action = rl.choose_action(state)
            # print(action.shape)
            noise = action_noise()
            action += noise
            collision = False
            st += 1

            next_state, dis, done, info = env.joints_step(action)
            collision = env.check_collision()
            # next_state, dis, done, _ = env.joints_step(action)
            if done:
                success = True

            a = np.sqrt(np.sum(np.square(action))).copy()
            # print(collision)
            r = -c1 * huber_loss(dis) - c2 * a - c3 * int(collision)
            rw += r
            if st == MAX_EP_STEPS:
                done = True
            rl.store_transition(state, action, r, next_state, done)
            state = next_state

            # 复原
            # if collision:
            #     env.move_object(env.robot.ur5, [0, 0, 0], [0, 0, 0, 1])
                # a=1

            if rl.memory_counter > MEMORY_CAPACITY:
                p.stepSimulation()
                rl.learn()
                var *= .9995
            if done:
                print("Episode {0}, Step:{1}, total reward:{2}, average reward:{3},{4}".format(i, st, rw, rw * 1.0 / st,
                                                                                               'success' if success else '----'))
                writer.add_scalar('reward/total_reward', rw, i)
                writer.add_scalar('reward/avg_reward', rw * 1.0 / st, i)
                total_rewards.append(rw)
                step_sums.append(st)
                break
        if rl.memory_counter > MEMORY_CAPACITY and i % 100 == 0:
        # if i%100 == 0:
            print("======================test mode===========================")
            acc = 0
            # test
            for j in range(100):
                obs, obs_flag = env.reset(show_debug)
                while not obs_flag:
                    obs, obs_flag = env.reset(show_debug)

                action_noise.reset()
                # obs = env.fix_target(box)
                state = obs.copy()
                test_reward = 0
                st = 0
                success = False
                while True:

                    p.stepSimulation()
                    state = np.array(state)

                    action = rl.choose_action(state)
                    st += 1

                    next_state, dis, done, info = env.joints_step(action)
                    collision = env.check_collision()
                    a = np.sqrt(np.sum(np.square(action))).copy()
                    # print(collision)
                    r = -c1 * huber_loss(dis) - c2 * a - c3 * int(collision)
                    test_reward += r
                    state = next_state
                    # if collision:
                    #     done = False
                        # env.robot.set_joints_state(env.robot.homej)
                        # env.move_object(env.robot.ur5, [0, 0, 0], [0, 0, 0, 1])

                    if done:
                        success = True

                    if st == MAX_EP_STEPS:
                        done = True

                    if success:
                        acc += 1

                    if done:
                        print("In test mode, Episode {0}, Step:{1}, total reward:{2}, average reward:{3},{4}".format(j,
                                                                                                                     st,
                                                                                                                     test_reward,
                                                                                                                     test_reward * 1.0 / st,
                                                                                                                     'success' if success else '----'))
                        writer.add_scalar('test_mode/total_reward', test_reward, int(i / 100 + 0.5) * 100 + j)
                        writer.add_scalar('test_mode/avg_reward', test_reward * 1.0 / st, int(i / 100 + 0.5) * 100 + j)
                        break
            print("test episode {0}, the accuracy is: {1}%".format(int(i / 100 + 0.5), acc))
            acc_epi.append(acc)
            writer.add_scalar('accuracy/per_100_episode', acc, int(i / 100 + 0.5))
            if acc >= last_max:
                last_max = acc
                rl.save_mode()
            acc = 0
    plot(MAX_EPISODES, total_rewards, acc_epi)

import DRL.DDPG as parm
import numpy as np
import datetime
import torch
import math
# from torch.utils.tensorboard import SummaryWriter
import pybullet as p
from ur5_envs.gym_ur5 import gym_ur5
from ur5_envs import models
from DRL.DDPG import DDPG
from DRL.OUnoise import OrnsteinUhlenbeckActionNoise
import mediapipe as mp
from PIL import Image
import cv2
import time
MAX_EPISODES = 8000
MAX_EP_STEPS = 150
MEMORY_CAPACITY = 100000
c1 = 1000
c2 = 0
c3 = 20
c4 = 100
def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5 * dis * dis
    else:
        return delta * (dis - 0.5 * delta)

print('runs/{}_DDPG_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "ur5_pybullet"))
env = gym_ur5(models.get_data_path(), disp=True)

show_debug = False
rl = DDPG()
rl.load_mode()
var = 0.3
total_rewards = []
step_sums = []
acc_epi = []
acc = 0
last_max = -1
action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(parm.ENV_PARAMS['action']), 0.05)

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
    mp_drawing = mp.solutions.drawing_utils
    mp_objectron = mp.solutions.objectron

    with mp_objectron.Objectron(static_image_mode=True,
                                max_num_objects=1,
                                min_detection_confidence=0.1,
                                model_name='Cup') as objectron1:
        with mp_objectron.Objectron(static_image_mode=True,
                                    max_num_objects=3,
                                    min_detection_confidence=0.1,
                                        model_name='Chair') as objectron:
            while True:
                p.stepSimulation()
                state = np.array(state)
                p.stepSimulation()

                viewMatrix = p.computeViewMatrix([2, 0, 2],
                                                 [0, 0, 0],
                                                 [0, 0, 1]
                                                 )
                # print(viewMatrix)
                projectionMatrix = p.computeProjectionMatrixFOV(55, 1, 0.01, 10, 0)

                _w, _h, RGB_Pixels, deph_Pixels, seg_Pixels = p.getCameraImage(700, 700, viewMatrix=viewMatrix,
                                                                               projectionMatrix=projectionMatrix)

                image = Image.fromarray(RGB_Pixels)
                # image.show()
                # a=1
                image = image.convert("RGB")
                # image.show()
                image = np.array(image)
                results = objectron1.process(image)
                # Draw box landmarks.
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(
                            image, detected_object.landmarks_2d,mp_objectron.BOX_CONNECTIONS,landmark_drawing_spec = None)  # 画框
                        # mp_drawing.draw_axis(image, detected_object.rotation,
                        #                      detected_object.translation)  # 画轴

                results = objectron.process(image)
                # Draw box landmarks.
                if results.detected_objects:
                    # print('Chair:')
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(
                            image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS,landmark_drawing_spec = None)  # 画框
                        # mp_drawing.draw_axis(image, detected_object.rotation,
                        #                      detected_object.translation)  # 画轴
                image = np.array(image)
                # Image.Image.show(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('3D box',image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            # cv2.imshow('MediaPipe Objectron',image)

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
                    print('test_mode/total_reward', test_reward, int(j + 0.5) * 100 + j)
                    print('test_mode/avg_reward', test_reward * 1.0 / st, int(j + 0.5) * 100 + j)
                    break
print("test episode {0}, the accuracy is: {1}%".format(int(j + 0.5), acc))

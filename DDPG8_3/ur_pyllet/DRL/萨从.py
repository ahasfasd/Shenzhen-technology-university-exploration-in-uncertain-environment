'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import math
import random

import gym
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
from IPython.display import clear_output
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if is_ipython:
    from IPython import display
import argparse
import time

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
writer = SummaryWriter('log/{}_SAC_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "Obstacle_avoidance"))
updates = 0
args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()


class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        global updates
        updates = updates + 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )
        writer.add_scalar("loss/q_value_loss1", q_value_loss1, updates)
        writer.add_scalar("loss/q_value_loss2", q_value_loss2, updates)
        writer.add_scalar("loss/policy_loss", policy_loss, updates)
        writer.add_scalar("loss/alpha", self.alpha, updates)
        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    # plt.show()

def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5*dis*dis
    else:
        return delta*(dis - 0.5*delta)

def CheckCollision(env):
    for i in range(env.sim.data.ncon):  # 表示有多少个接触力
        contact = env.sim.data.contact[i]  # 接触力的数据
        # 输出当前状态下每一个接触力对应的是那两个geom  输出名字
        name = ['obstacle', 'table', 'robot0']
        can_colission = ['robot0:base_link']
        links_collission =  ['robot0:torso_lift_link', 'robot0:head_pan_link', 'robot0:head_tilt_link',
                     'robot0:head_camera_link', 'robot0:shoulder_pan_link', 'robot0:shoulder_lift_link',
                     'robot0:upperarm_roll_link', 'robot0:elbow_flex_link', 'robot0:forearm_roll_link',
                    'robot0:wrist_flex_link', 'robot0:wrist_roll_link', 'robot0:gripper_link', 'robot0:r_gripper_finger_link',
                    'robot0:l_gripper_finger_link', 'obstacle', 'table']
        vis = False
        # print(i)
        str1 = env.sim.model.geom_id2name(contact.geom1)
        str2 = env.sim.model.geom_id2name(contact.geom2)
        # print(str1)
        # print(str2)
        # print("\n")
        for j in can_colission:
            if str1.find(j) >=0 or str2.find(j) >=0:
                vis = True
        if vis:
            continue
        vis = False
        for j in range(len(links_collission)):
            for k in range(len(links_collission)):
                if (j == 14 and k == 15) or (j == 15 and k == 14):
                    continue
                if str1.find(links_collission[j]) >= 0 and str2.find(links_collission[k]) >= 0:
                    # vis = True
                    # print('geom1', contact.geom1, str1)
                    # print('geom2', contact.geom2, str2)
                    return True # 不允许自碰撞
        # if vishuber_loss: # 允许自碰撞
        #     continue
            # 如果你输出的name是None的话  请检查xml文件中的geom元素是否有名字着一个属性
            # print("wrong")
    return False


def check_space(p):
    if p[0] >= 0.8 and p[0] <= 1.8 and p[1]>=0.2 and p[1]<=1.2 and p[2] >= 0.4 and p[2] <= 1.3:
        return True
    return False

def plot_graph(episode, rewards):
    clear_output(True)
    plt.figure(2)
    plt.clf()
    plt.title('frame %s. reward: %s' % (episode, rewards[-1]))
    plt.xlabel('Episode')
    plt.ylabel('Total_Rewards')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

replay_buffer_size = 50000
replay_buffer = ReplayBuffer(replay_buffer_size)

ENV = ""
SCREEN_SHOT = False
SPARSE_REWARD = False
# choose env
# ENV = ['Reacher', 'Pendulum-v0', 'HalfCheetah-v2'][2]
# if ENV == 'Reacher':
#     NUM_JOINTS = 2
#     LINK_LENGTH = [200, 140]
#     INI_JOING_ANGLES = [0.1, 0.1]
#     SCREEN_SIZE = 1000
#     SPARSE_REWARD = False
#     SCREEN_SHOT = False
#     action_range = 10.0
#     env = Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths=LINK_LENGTH, \
#                   ini_joint_angles=INI_JOING_ANGLES, target_pos=[369, 430], render=True, change_goal=False)
#     action_dim = env.num_actions
#     state_dim = env.num_observations
# else:
#     env = NormalizedActions(gym.make(ENV))
#     action_dim = env.action_space.shape[0]
#     state_dim = env.observation_space.shape[0]
#     action_range = 1.

action_dim = 7
state_dim = 28
action_range = 1.

# hyper-parameters for RL training
max_episodes = 8000
max_steps = 20 if ENV == 'Reacher' else 200  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx = 0
batch_size = 128
explore_steps = 20000  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 512
rewards = []
model_path = './model/sac_v2'
c1 = 2000
c2 = 200
c3 = 120
c4 = 50
sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

if __name__ == '__main__':
    if args.train:
        # training loop
        ep_rs = []
        root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
        object_pos = [1.5, 0.25, 0.45, 1.5, 0.75, 0.5, 1.5, 1.25, 0.5]
        object_size = [0.05, 0.03, 0.43, 0.04, 0.04, 0.65, 0.04, 0.07, 0.71]
        for eps in range(max_episodes):
            s = env.reset()
            env.test_set_joint(root)
            box_position = s['desired_goal']
            ep_r = 0

            success = False
            for step in range(max_steps):
                env.render()
                joints = env.test_get_joint()
                state = np.array([])
                state = np.append(state, joints)
                state = np.append(state, box_position)
                state = np.append(state, object_pos)
                state = np.append(state, object_size)
                if frame_idx > explore_steps:
                    action = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
                else:
                    action = sac_trainer.policy_net.sample_action()
                if ENV == 'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    for i in range(10):
                        next_state, _, done, info = env.test_step(action / 10)
                        env.render()
                        collision = CheckCollision(env=env)
                        if collision:
                            next_state, _, done, info = env.test_step(-action * 2 / 10)
                        if not check_space(next_state['achieved_goal']):
                            # print(next_state['achieved_goal'])
                            next_state, _, done, info = env.test_step(-action * 2 / 10)
                    # env.render()

                d = np.sqrt(np.sum(np.square(box_position - next_state['achieved_goal']))).copy()  # L2 distance
                dis = (np.sum(np.square(box_position - next_state['achieved_goal']))) / 7  # MSE
                # dis = (np.sum(np.abs(box_position - next_state['achieved_goal'])))/7                  # MAE
                a = np.sqrt(np.sum(np.square(action))).copy()
                joints = env.test_get_joint()
                next_state = np.array([])
                next_state = np.append(next_state, joints)
                next_state = np.append(next_state, box_position)
                next_state = np.append(next_state, object_pos)
                next_state = np.append(next_state, object_size)
                if d <= 0.05 and not collision:
                    success = True
                r = -c1 * huber_loss(d) - c2 * a - c3 * int(collision)
                ep_r += r

                done = 1 if success or i == 199 else 0
                replay_buffer.push(state, action, r, next_state, done)

                state = next_state
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                               target_entropy=-1. * action_dim)

                if done:
                    break
            writer.add_scalar("ave_reward/train_reward",ep_r, eps)
            print('Ep: ', eps,
                  '| Ep_r: ', round(ep_r, 2),
                  "success" if success else "-------")
            ep_rs.append(ep_r)
            plot_graph(eps, ep_rs)
        clear_output(True)
        plt.figure(2)
        plt.clf()
        plt.xlabel('Episode')
        plt.ylabel('Total_Rewards')
        plt.plot(ep_rs)
        plt.show()
        plt.savefig('SAC_result')
        np.save("SAC_reward", ep_rs)

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(10):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            else:
                state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
                if ENV == 'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                else:
                    next_state, reward, done, _ = env.step(action)
                    env.render()

                episode_reward += reward
                state = next_state

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
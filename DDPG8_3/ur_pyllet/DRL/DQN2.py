"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from robotics.fetch.reach import FetchReachEnv
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import clear_output
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.00005                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 50000
EPISODE = 8000
env = FetchReachEnv(model_name="Three_obstacle")


N_ACTIONS = 14
N_STATES = 7 + 3 + 18
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

c1 = 2000
c2 = 200
c3 = 120
c4 = 50
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
        # if vis: # 允许自碰撞
        #     continue
            # 如果你输出的name是None的话  请检查xml文件中的geom元素是否有名字着一个属性
            # print("wrong")
    return False

def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5*dis*dis
    else:
        return delta*(dis - 0.5*delta)

def check_space(p):
    if p[0] >= 0.8 and p[0] <= 1.8 and p[1]>=0.2 and p[1]<=1.2 and p[2] >= 0.4 and p[2] <= 1.3:
        return True
    return False


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(512, 1024)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(1024, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, episode):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < min(0.95, (1-np.exp(-episode/1500))):   # greedy
        # if np.random.uniform() < EPISODE:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] #if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model_dict(self, model_name):
        self.eval_net.load_state_dict(torch.load("model/" + model_name + "dqn_eval.pkl"))
        self.target_net.load_state_dict(torch.load("model/" + model_name + "dqn_target.pkl"))

    def save_model_dict(self, model_name):
        torch.save(self.eval_net.state_dict(), "model/" + model_name + "dqn_eval.pkl")
        torch.save(self.target_net.state_dict(), "model/" + model_name + "dqn_target.pkl")


def plot_graph(episode, rewards):
    clear_output(True)
    plt.figure(1)
    plt.clf()
    plt.title('frame %s. reward: %s' % (episode, rewards[-1]))
    plt.xlabel('Episode')
    plt.ylabel('Total_Rewards')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

dqn = DQN()

print('\nCollecting experience...')
maxr = -9999999
ep_rs = []
root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
object_pos = [1.5, 0.25, 0.45, 1.5, 0.75, 0.5, 1.5, 1.25, 0.5]
object_size = [0.05, 0.03, 0.43, 0.04, 0.04, 0.65, 0.04, 0.07, 0.71]
for i_episode in range(EPISODE):
    s = env.reset()
    box_position = s['desired_goal']
    ep_r = 0

    env.test_set_joint(root)
    success = False
    for i in range(200):
        env.render()
        joints = env.test_get_joint()
        state = np.array([])
        state = np.append(state, joints)
        state = np.append(state, box_position)
        state = np.append(state, object_pos)
        state = np.append(state, object_size)

        a = dqn.choose_action(state, i_episode)

        # take action
        s_, r, done, info = env.SparseStep(a)
        ac = 1 if a % 2 == 0 else -1
        collision = CheckCollision(env=env)
        if collision:
            s_, r, done, info = env.SparseStep(a + ac)
            s_, r, done, info = env.SparseStep(a + ac)
        if not check_space(s_['achieved_goal']):
            # print(next_state['achieved_goal'])
            s_, r, done, info = env.SparseStep(a + ac)
            s_, r, done, info = env.SparseStep(a + ac)

        act = np.pi*5/360
        joints = env.test_get_joint()
        next_state = np.array([])
        next_state = np.append(next_state, joints)
        next_state = np.append(next_state, box_position)
        next_state = np.append(next_state, object_pos)
        next_state = np.append(next_state, object_size)

        # modify the reward

        dis = np.sqrt(np.sum(np.square(box_position - s_['achieved_goal']))).copy()  # L2 distance
        if dis <= 0.05 and not collision:
            done = True
            success = True

        r = -c1 * huber_loss(dis) - c2*act -c3 * int(collision)

        dqn.store_transition(state, a, r, next_state)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break

        if collision:
            env.test_set_joint(root)
        s = s_
        state = next_state
    print('Ep: ', i_episode,
          '| Ep_r: ', round(ep_r, 2),
          "success" if success else "-------")
    ep_rs.append(ep_r)
    plot_graph(i, ep_rs)
    if maxr < ep_r:
        maxr = ep_r
        dqn.save_model_dict(model_name='Three_obstacle')
clear_output(True)
plt.figure(1)
plt.clf()
plt.xlabel('Episode')
plt.ylabel('Total_Rewards')
plt.plot(ep_rs)
plt.show()
plt.savefig('DQN_result2')
np.save("DQN_reward2", ep_rs)
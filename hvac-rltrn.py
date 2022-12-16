import math, random, os, csv
from selectors import EpollSelector
import socket
import subprocess as sp
from turtle import color
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import data_loader as DL
import json
import matplotlib.pyplot as plt
from collections import deque

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
Variable = lambda *args: args[0].clone().detach().cuda() if USE_CUDA else args[0].clone().detach()
config = json.load(open('../config.json', 'r'))

HOST = '127.0.0.1'
PORT = 9999
bestPWH = 10000000000
name = "jun"
num_data = DL.data_dict[name]["data_num"]
data_name = DL.data_dict[name]['data_name']
train_episode = DL.data_dict[name]['trains']

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))

csv_f = open('csv_dqn/pyFile.csv', 'a', newline='')
csv_file = csv.writer(csv_f)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon, test=False):
        if random.random() > epsilon or test:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(dim=1)[1].data[0]
        else:
            action = random.randrange(2)
        return action

class ENV():
    def __init__(self):
        self.state = [0., 0., 0., 0., 0., 0.]
        # 위도|경도|실내온도|실외온도|시간|전력
        self.line = 0
        self.SumPWH = 0
        self.preaction = 0
        self.idx = 2

    def step(self, action):
        current_t, next_t, lat, lng, done, _ = DL.get_data(self.line, self.idx, data_name)
        self.idx += 1
        server_socket.listen()
        client_socket, addr = server_socket.accept()
        while True:
            data = client_socket.recv(1024)
            if data:
                data = data.decode()
                data = data.split(',')
                self.state[0] = round(lat, 2)
                self.state[1] = round(lng, 2)
                self.state[2] = round((float(data[0]) - 22) / (32 - 22), 1)
                self.state[3] = round((float(data[5]) - 22) / (32 - 22), 1)
                self.state[4] = round(current_t, 2)
                self.state[5] = round(self.SumPWH / 10000000, 2)
                self.SumPWH = float(data[3])
                real_temp = float(data[0])
                print(self.state)
                #print(data[4])
                client_socket.sendall('{},{}'.format(action, next_t).encode())
                break
            else:
                print("No Signal")
        reward = DL.get_reward(self.state[2], done, self.SumPWH, bestPWH, self.preaction, action)
        print("Action :", action, " | Reward :", reward)
        self.preaction = action
        return self.state, reward, done, real_temp
    def reset(self):
        self.state = [0., 0., 0., 0., 0., 0.]
        self.SumPWH = 0
        self.preaction = 0
        self.idx = 3
        return self.state

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = policy_net(state)
    next_q_values = torch.zeros(batch_size)
    next_q_values = target_net(next_state)
    
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # action = behavior policy

    next_q_value = next_q_values.max(1)[0]

    

    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (Variable(expected_q_value.data) - q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

policy_net = DQN(6, 2)
target_net = DQN(6, 2)
target_net.load_state_dict(policy_net.state_dict())

target_net.train()

env = ENV()

if USE_CUDA:
    policy_net = policy_net.cuda()
    target_net = target_net.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
episode_step = 0

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

episode_reward = 0
optimizer = optim.Adam(policy_net.parameters())
replay_buffer = ReplayBuffer(10000)

EPI_NUM = 1000
TARGET_UPDATE = 10
batch_size = 128
gamma = 0.99
frame_idx = 0
start_time = time.time()
loss_list = []
pwh_list = []
arrival_temp = []
avg_pwh_list = []
pwh_sum = 0

state = [0, 0, 0, 0, 0, 0]

temp_min = np.zeros(EPI_NUM)+24
temp_max = np.zeros(EPI_NUM)+26
done = False

for iteration in range(EPI_NUM):
    while(not done):
        epsilon = epsilon_by_frame(frame_idx)
        action = policy_net.act(state, epsilon)
        if (str(type(action)) != "<class 'int'>"):
            action = action.item()
        next_state, reward, done, real_temp = env.step(action)
        
        if env.idx>3:
            replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        if done:
            arrival_temp.append(real_temp)
            pwh_list.append(env.SumPWH)
            pwh_sum += env.SumPWH
            if iteration%TARGET_UPDATE==0:
                target_net.load_state_dict(policy_net.state_dict())
            end_time = time.time()
            if env.line < num_data-1:
                env.line += 1
            else:
                env.line = 0
            
            if env.state[2]<0.5:
                if iteration%TARGET_UPDATE==0:
                    avg_pwh_list.append(pwh_sum/TARGET_UPDATE)
                    pwh_sum = 0
                if bestPWH > env.SumPWH:
                    bestPWH = env.SumPWH
                    torch.save(policy_net.state_dict(), 'saved_models/dqn/useGPS/bestmodelUseHuber-{}-gps.pt'.format(data_name))
            torch.save(policy_net.state_dict(), 'saved_models/dqn/useGPS/currentmodelUseHuber-{}-gps.pt'.format(data_name))

            

            print("========================= {} STEP FINISHED =========================".format(iteration))
            print("step delay :", end_time - start_time)
            print("REWATD: {}".format(episode_reward))
            print("ENERGY: {}".format(int(env.SumPWH)))
            print("BEST ENERGY: {}".format(int(bestPWH)))
            state = env.reset()
            episode_reward = 0
            start_time = time.time()
        
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            loss_list.append(loss.item())
        frame_idx+=1
    
    done = False


print(avg_pwh_list)
csv_file.writerow(pwh_list)
csv_f.close()
plt.subplot(311)
plt.plot(loss_list, color='red', )
plt.xlabel("compute step")
plt.ylabel("loss")

plt.subplot(312)
plt.plot(pwh_list, color='blue')
plt.xlabel("episode")
plt.ylabel("PWH")

plt.subplot(313)
plt.plot(arrival_temp, color='green')
plt.plot(temp_min, color='red')
plt.plot(temp_max, color='red')
plt.xlabel("episode")
plt.ylabel("temp")

plt.show()

plt.plot(avg_pwh_list)
plt.show()


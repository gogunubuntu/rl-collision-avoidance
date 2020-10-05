import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from collections import deque
import random 

def print_shape(sequence, X) : 
    print(X, np.asarray(sequence).shape)

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


class replay_buffer() : 
    def __init__(self, buff_limit) : 
        self.buffer = deque(maxlen = buff_limit)
    def put(self, memory) :
        for e in memory : 
            for i in range(len(e[0])):
                temp = []
                for j in range(5) : 
                    temp.append(e[j][i])
                self.buffer.append(temp)
    def print_elements(self) : 
        print(np.asarray(self.buffer).shape)
    
    def sample(self, num_sampling):
        transitions = random.sample(self.buffer, num_sampling)
        a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], []
        obs_lst, goal_lst, speed_lst = [], [], []
        obs_prime_lst, goal_prime_lst, speed_prime_lst = [], [], []

        for transition in transitions : 
            s, a, r, s_prime, done_mask = transition

            
            obs_lst.append(s[0])
            goal_lst.append(s[1])
            speed_lst.append(s[2])

            a_lst.append([a])
            r_lst.append([r])
             
            obs_prime_lst.append(s_prime[0])
            goal_prime_lst.append(s_prime[1])
            speed_prime_lst.append(s_prime[2])

            done_mask_lst.append([done_mask])

        return np.asarray(obs_lst),       np.asarray(goal_lst),       np.asarray(speed_lst),\
               np.asarray(a_lst),         np.asarray(r_lst), \
               np.asarray(obs_prime_lst), np.asarray(goal_prime_lst), np.asarray(speed_prime_lst), \
               np.asarray(done_mask_lst)
    def size(self) : 
        return len(self.buffer)

def generate_action(env, state_list, qnet, epsilon):
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        a = qnet.sample_action(s_list, goal_list, speed_list, epsilon)
        a = np.array(a)
    else:
        a = None
    return a


def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns

def dqn_update_stage1(qnet, qnet_target, memory, batch_size, optimizer, epoch, gamma) : 
    for update in range(epoch):
        obs_batch, goal_batch, speed_batch, a_batch, r_batch, obs_prime_batch, goal_prime_batch, speed_prime_batch, done_mask_batch = memory.sample(num_sampling=batch_size)

        obs_batch = Variable(torch.from_numpy(obs_batch)).float().cuda()
        goal_batch = Variable(torch.from_numpy(goal_batch)).float().cuda()
        speed_batch = Variable(torch.from_numpy(speed_batch)).float().cuda()

        obs_prime_batch = Variable(torch.from_numpy(obs_prime_batch)).float().cuda()
        goal_prime_batch = Variable(torch.from_numpy(goal_prime_batch)).float().cuda()
        speed_prime_batch = Variable(torch.from_numpy(speed_prime_batch)).float().cuda()

        a_batch = Variable(torch.from_numpy(a_batch)).long().cuda()
        r_batch = Variable(torch.from_numpy(r_batch)).float().cuda()
        done_mask_batch = Variable(torch.from_numpy(done_mask_batch)).float().cuda()

        q_out = qnet(obs_batch, goal_batch, speed_batch)

        q_a = q_out.gather(1,a_batch)
        
        max_q_prime = qnet_target(obs_prime_batch, goal_prime_batch, speed_prime_batch).max(1)[0].unsqueeze(1)
        target = r_batch + gamma * max_q_prime * done_mask_batch
        
        loss = F.mse_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





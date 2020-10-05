import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model.net import CNNqnet
from stage_world1 import StageWorld

from model.dqn import dqn_update_stage1
from model.dqn import replay_buffer
from model.dqn import generate_action

MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 64
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 10
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5
BUFFER_LIMIT = 1e5
ACTION_DIM = 30
LEARNING_START = 5000
TARGET_UPDATE_INTERVAL = 500
EPSILON_DECAY = 1000
def run(comm, env, qnet, qnet_target, policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)
    buff = []
    memory = replay_buffer(buff_limit=BUFFER_LIMIT)
    global_update = 0
    global_step = 0

    
    if env.index == 0:
        env.reset_world()

    for id in range(MAX_EPISODES):
        env.reset_pose()
        epsilon = max(0.01, 0.08 - 0.01*(MAX_EPISODES/EPSILON_DECAY))
        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        while not terminal and not rospy.is_shutdown():
            ##s
            state_list = comm.gather(state, root=0)
            a = generate_action(env=env, state_list=state_list, 
                                qnet=qnet, epsilon=epsilon)
            #print(a)
            if env.index == 0 : 
                temp = np.array(a, dtype=np.float32)
                temp = temp/(ACTION_DIM-1)*2 -1
                scaled_action = np.ones((a.shape[0], 2))
                scaled_action[:, 1] = temp
                #print(scaled_action)
            else : 
                scaled_action = None

            real_action = comm.scatter(scaled_action, root=0)
            env.control_vel(real_action)

            # rate.sleep()
            rospy.sleep(0.001)

            r, terminal, result = env.get_reward_and_terminate(step)
           
            if terminal : 
                done_mask = 0
            else : 
                done_mask = 1
            
            ep_reward += r
            global_step += 1


            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            ## r, done
            r_list = comm.gather(r, root=0)
            done_mask_list = comm.gather(done_mask, root=0)
            ##s'
            state_next_list = comm.gather(state_next, root=0)
            if env.index == 0:
                buff.append((state_list, a, r_list, state_next_list, done_mask_list))
                memory.put(buff)
                buff = []
                if memory.size() > LEARNING_START:
                    dqn_update_stage1(qnet=qnet, qnet_target=qnet_target, memory=memory, batch_size=BATCH_SIZE, optimizer=optimizer, epoch=EPOCH,  gamma=GAMMA)
                    global_update += 1
                
                if global_update != 0 and global_update % TARGET_UPDATE_INTERVAL == 0:
                    torch.save(qnet.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                    logger.info('########################## model saved when update {} times#########'
                                '################'.format(global_update))
                    qnet_target.load_state_dict(qnet.state_dict())
            step += 1
            state = state_next
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))

if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'q'
        # policy = MLPPolicy(obs_size, act_size)
        ## policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        qnet = CNNqnet(frames=LASER_HIST, out_dim=ACTION_DIM)
        
        qnet_target = CNNqnet(frames=LASER_HIST, out_dim=ACTION_DIM)
        qnet_target.load_state_dict(qnet.state_dict())
        
        ## policy.cuda()
        qnet.cuda()
        qnet_target.cuda()
        
        ##opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        opt = Adam(qnet.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/201001'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            qnet.load_state_dict(state_dict)
            qnet_target.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        #policy = None
        qnet_target = None
        qnet = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, qnet=qnet, qnet_target=qnet_target, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass

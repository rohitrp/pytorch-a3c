from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
from torch.nn.init import xavier_uniform_ as Xavier

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--tag', required=True,
                    help='tag for current experiment')
parser.add_argument('--transfer-env', required=True,
                    help='env model to use for initializing the network')
parser.add_argument('--transfer-tag', required=True,
                    help='tag for transfer env model')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transfer_env = create_atari_env(args.transfer_env)
    transfer_env_model = ActorCritic(
        transfer_env.observation_space.shape[0], transfer_env.action_space
    )

    checkpoint = torch.load('./checkpoints/{}_{}.tar'.format(args.transfer_env, args.transfer_tag))

    try:
        transfer_env_model.load_state_dict(checkpoint['model'])
    except Exception:
        transfer_env_model.load_state_dict(checkpoint['shared_model'])

    transfer_env_model.train()

    env = create_atari_env(args.env_name)

    # shared_model = nn.Sequential(*list(transfer_env_model.children())[:-1])
    # shared_model.add_module('actor_linear', nn.Linear(256, env.action_space.n))

    shared_model = transfer_env_model
    shared_model.actor_linear = nn.Linear(256, env.action_space.n)
    shared_model.share_memory()

    tag = args.tag

    if tag == 'all_train':
        from configs.all_train import Config
    elif tag == 'train_conv4':
        from configs.train_conv4 import Config
    elif tag == 'reset_conv4_lstm_fc':
        from configs.reset_conv4_lstm_fc import Config
    elif tag == 'reset_lstm_fc':
        from configs.reset_lstm_fc import Config
    else:
        raise 'Invalid config'
    
    config = Config()

    # Freeze layers based on config values
    for parameter in shared_model.conv1.parameters():
        parameter.requires_grad = config.conv1_train
    
    for parameter in shared_model.conv2.parameters():
        parameter.requires_grad = config.conv2_train
    
    for parameter in shared_model.conv3.parameters():
        parameter.requires_grad = config.conv3_train
    
    for parameter in shared_model.conv4.parameters():
        parameter.requires_grad = config.conv4_train
    
    for parameter in shared_model.lstm.parameters():
        parameter.requires_grad = config.lstm_train
    
    for parameter in shared_model.critic_linear.parameters():
        parameter.requires_grad = config.critic_linear_train
    
    for parameter in shared_model.actor_linear.parameters():
        parameter.requires_grad = config.actor_linear_train


    if(config.conv1_reset==True):
        Xavier(shared_model.conv1.weight)
        shared_model.conv1.bias.data.fill_(0.01)

    if(config.conv2_reset==True):
        Xavier(shared_model.conv2.weight)
        shared_model.conv2.bias.data.fill_(0.01)

    if(config.conv3_reset==True):
        Xavier(shared_model.conv3.weight)
        shared_model.conv3.bias.data.fill_(0.01)

    if(config.conv4_reset==True):
        Xavier(shared_model.conv4.weight)
        shared_model.conv4.bias.data.fill_(0.01)

    if(config.lstm_reset==True):
        #Xavier(shared_model.lstm.weight)
        #shared_model.lstm.bias.data.fill_(0.01)
        for name, param in shared_model.lstm.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
    
    if(config.critic_linear_reset==True):
        Xavier(shared_model.critic_linear.weight)
        shared_model.critic_linear.bias.data.fill_(0.01)

    if(config.actor_linear_reset==True):
        Xavier(shared_model.actor_linear.weight)
        shared_model.actor_linear.bias.data.fill_(0.01)
    

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes,
                                      args, shared_model, counter, optimizer))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

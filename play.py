import gym
import argparse
import torch
import torch.nn.functional as F

from model import ActorCritic
import time
from envs import create_atari_env

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Env name', required=True)
parser.add_argument('--tag', help='Experiment tag', required=True)
parser.add_argument('--render', help='Enable rendering', action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filename = './checkpoints/{}_{}.tar'.format(args.env, args.tag)
checkpoint = torch.load(filename, map_location=device)

env = create_atari_env(args.env)

model = ActorCritic(env.observation_space.shape[0], env.action_space)
try:
  model.load_state_dict(checkpoint['model'])
except Exception:
  model.load_state_dict(checkpoint['shared_model'])

model.eval()

state = env.reset()
state = torch.from_numpy(state)
reward_sum = 0
done = True

while True:
  if args.render: 
    env.render()
  
  if done:
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)
  else:
    cx = cx.detach()
    hx = hx.detach()

  with torch.no_grad():
    value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
  prob = F.softmax(logit, dim=-1)
  action = prob.max(1, keepdim=True)[1].numpy()

  state, reward, done, _ = env.step(action[0, 0])
  reward_sum += reward

  if done:
    print('Rewards', reward_sum)
    env.close()
    break
  
  state = torch.from_numpy(state)
  
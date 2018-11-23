import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import os

def test(rank, args, shared_model, counter, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    episode_rewards = []
    episode_durations = []
    episode_lengths = []
    
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
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
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            time_elapsed = time.gmtime(time.time() - start_time)
            episode_rewards.append(reward_sum)
            episode_durations.append(time_elapsed)
            episode_lengths.append(episode_length)
            
            checkpoint_filename = './checkpoints/{}_{}.tar'.format(args.env_name, args.tag)

            os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_durations': episode_durations,
                'episode_lengths': episode_lengths
            }, '{}'.format(checkpoint_filename))

            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time_elapsed),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(5)

        state = torch.from_numpy(state)

import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.style.use('seaborn-darkgrid')
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import argparse
from time import mktime
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='Path to checkpoint', required=True)

args = parser.parse_args()
plot_filename = os.path.splitext(os.path.basename(args.checkpoint))[0]

checkpoint = torch.load(args.checkpoint)
episode_rewards = checkpoint['episode_rewards']
episode_durations = checkpoint['episode_durations']
episode_lengths = checkpoint['episode_lengths']

episode_durations = [datetime.fromtimestamp(mktime(t)) for t in episode_durations]

fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(3,1, hspace=0.5)
ax = pl.subplot(gs[0, 0])
ax.plot(episode_durations, episode_rewards)
ax.set_title('Time')
ax.set_xlabel('Time')
ax.set_ylabel('Reward')

plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Mh %Mm %Ss')
plt.gca().xaxis.set_major_formatter(myFmt)

ax = pl.subplot(gs[1, 0])
ax.plot(episode_rewards)
ax.set_title('Rewards')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')

ax = pl.subplot(gs[2, 0])
ax.plot(episode_lengths)
ax.set_title('Timesteps')
ax.set_xlabel('Episode')
ax.set_ylabel('Timestep')

plot_filename = './plots/{}.png'.format(plot_filename)
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

plt.savefig('{}'.format(plot_filename))
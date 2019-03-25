import argparse
import math
import os
import random
import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from lib.common import mkdir
from lib.model import ActorCritic
from lib.multiprocessing_env import SubprocVecEnv


NUM_ENVS            = 8 #num of parallel envs
ENV_ID              = "Humanoid-v2"
HIDDEN_SIZE         = 64
LEARNING_RATE       = 1e-3
GAMMA               = 0.99 #discount factor
GAE_LAMBDA          = 0.95 #smoothing factor
PPO_EPSILON         = 0.2 #clip of the ratio
CRITIC_DISCOUNT     = 0.5 # loss tends be bigger than actor, so we scale it down
ENTROPY_BETA        = 0.001 # the amount of imporatence to give to the entropy bonus which helps exploration
'''
# number of transitions we sample for each training iteration, each step
collects a transitoins from each parallel env, hence total amount of data
collected = N_envs * PPOsteps --> buffer of 2048 data samples to train on
'''
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64 # num of samples that are randomly  selected from the total amount of stored data
'''one epoch means one PPO-epochs -- one epoch means one pass over the entire buffer of training data.
So if one buffer has 2048 transitions and mini-batch-size is 64, then one epoch would be 32 selected mini batches.
'''
PPO_EPOCHS          = 10 # how many times we propagate the network over the entire buffer of training data
TEST_EPOCHS         = 10 # how often we run tests to eval our network, one epoch is one entire ppo update cycle
NUM_TESTS           = 10 # num of tests we run to average the total rewards, each time we want eval the performance of the network
TARGET_REWARD       = 500


def make_env():
    ''' returns a function which creates a single environment '''
    def _thunk():
        env = gym.make(ENV_ID)
        return env
    return _thunk


def test_env(env, model, device, deterministic=True):
    '''
    Training: sampling actions semi-randomly from the prob dist output by the network, so we get exploration
    Testing: deterministic not random

    functions runs for one episode and returns total reward
    '''
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        #continous action space instead of sampling based on the mean and stdf, we use means
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    '''
    mask is 0 if state is terminal, otherwise 1
    '''
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))): #looping backwards from last step from the most recent experience to earlier
        # bellman equaiton minus the value of the state and is essentially the same as the advantage
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] #if mask = 0, we use just reward as terminal state has no next_state
        gae = delta + gamma * lam * masks[step] * gae # moving average of advantages discounted by gamma * gae lambda
        # prepend to get correct order back
        returns.insert(0, gae + values[step]) # add the value of the state we subtracted back in
    return returns #ppo steps long list and env num wide


def ppo_iter(states, actions, log_probs, returns, advantage):
    '''generates random mini-batches until we have covered the full batch'''
    #if update batch contains 2048 trajectories, and MINI_BATCH_SIZE=64, then 32 mini batches per epoch
    batch_size = states.size(0)

    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):

            dist, value = model(state) #state into network to get latest prob dist and value of the state
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action) # with each succesive update

            # SURROGAGE POLICY LOSS in log space
            # A long trajectory of experiences is collected at each update cycle
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            #Mean squared error between the actual GAE returns
            #and network estimated value of the state

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
            #from paper


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy

            count_steps += 1

    writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)


if __name__ == "__main__":
    mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    args = parser.parse_args()
    writer = SummaryWriter(comment="ppo_" + args.name)

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # Prepare parallel nvironments
    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_ID)
    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("parameters:",model.parameters())

    frame_idx  = 0
    train_epoch = 0 #one complte update cycle
    best_reward = None

    state = envs.reset() # 8 actions, 8 next states, 8 rewards, and 8 dones
    early_stop = False

    while not early_stop:
        #storing training data
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in range(PPO_STEPS): #each ppo steps generates actions, states, rewards

            state = torch.FloatTensor(state).to(device)

            dist, value = model(state)

            action = dist.sample()
            # each state, reward, done is a list of results from each parallel environment
            next_state, reward, done, _ = envs.step(action.cpu().numpy()) # really a lists of state foe each env
            log_prob = dist.log_prob(action) #pass through the network

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))

            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)
            # storing, each list will be len(PPO_steps) and contains a 8 wide list
            state = next_state
            frame_idx += 1

        #run the final next state through the nework to get its value
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        #256*8 = 2048
        ### trajectory begin
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        ### trajectory end
        #subtract returns from the network estimated values of each state
        advantage = returns - values
        advantage = normalize(advantage) #helps smooth training

        ppo_update(frame_idx, states, actions, log_probs, returns, advantage)
        train_epoch += 1

        if train_epoch % TEST_EPOCHS == 0: #one test epoch is one entire update operation
            #every few epochs we run a series of tests and average the rewards to see the agents performance
            test_reward = np.mean([test_env(env, model, device) for _ in range(NUM_TESTS)])
            writer.add_scalar("test_rewards", test_reward, frame_idx)
            print('Frame %s. reward: %s' % (frame_idx, test_reward))
            # Save a checkpoint every time we achieve a best reward
            if best_reward is None or best_reward < test_reward:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
                    fname = os.path.join('.', 'checkpoints', name)
                    print(fname)
                    torch.save(model, fname)
                best_reward = test_reward
            if test_reward > TARGET_REWARD: early_stop = True

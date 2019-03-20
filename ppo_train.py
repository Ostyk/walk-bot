import argparse
import math
import os
import random
import gym
import numpy as np

from model import ActorCritic
import tensorflow as tf


NUM_ENVS            = 1
ENV_ID              = "Humanoid-v2"
#ENV_ID = "RoboschoolHumanoid-v1"

HIDDEN_SIZE         = 64
LEARNING_RATE       = 1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 2#56 # number of transitions we sample for each training iteration, each step collects a transitoins from each parallel env, hence total amount of data collected = N_envs * PPOsteps = buffer of 2048 data samples to train on
MINI_BATCH_SIZE     = 2 # num of samples that are randomly  selected from the total amount of stored data
PPO_EPOCHS          = 10 #
'''one epoch means one PPO-epochs -- one epochd means one pass over the entire buffer of training data.
So if one buffer has 2048 transitions and mini-batch-size is 64, then one epoch would be 32 selected mini batches.
'''
TEST_EPOCHS         = 10 # how often we run tests to eval our network, one epoch is one entire ppo update cycle
NUM_TESTS           = 10 #num of tests we run to average the total rewards, each time we want eval the performance of the network
TARGET_REWARD       = 1000

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] #basically the bellman equaiton minus the value of the step
        gae = delta + gamma * lam * masks[step] * gae #moving average
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns
def normalize(x):
    x -= tf.reduce_mean(x)
    x /= (tf.keras.backend.std(x) + 1e-8)
    return x

def ppo_iter(states, actions, log_probs, returns, advantage):
    #batch_size = states.size(0)
    #[print(i.shape) for i in [states, actions, log_probs, returns, advantage]]
    #batch_size = int(states.get_shape()[0])
    #print(batch_size//MINI_BATCH_SIZE)
    # generates random mini-batches until we have covered the full batch
    g = lambda x: tf.unstack(x)
    yield g(states), g(actions), g(log_probs), g(returns), g(advantage)
    #print(batch_size // MINI_BATCH_SIZE)
    #rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
    #print(rand_ids)
    #print(tf.slice(states, rand_ids))
    #for _ in range(batch_size // MINI_BATCH_SIZE):
    #   rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
    #   print("P")
    #   yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):

    count_steps, sum_returns, sum_advantage, sum_loss_actor =  0, 0.0, 0.0, 0.0
    sum_loss_critic, sum_entropy, sum_loss_total = 0.0, 0.0, 0.0


    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        #for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            state = state[0]
            action, value, norm_dist = model.act(sess.run(state))
            entropy = norm_dist.entropy()
            new_log_probs = norm_dist.log_prob(action)
            ratio = tf.exp(new_log_probs - old_log_probs)
            actor_loss = -tf.reduce_mean(tf.minimum(
                tf.multiply(ratio, advantage), #surrogate 1
                tf.clip_by_value(ratio, 1.-clip_param, 1.+clip_param)*advantage))

            critic_loss = tf.reduce_mean(tf.square(tf.subtract(return_, value)))
            print("made it")
            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy
            optimizer.minimize(loss)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            return_ = return_[0]
            # track statistics
            sum_returns += tf.reduce_mean(return_)
            sum_advantage += tf.reduce_mean(advantage)
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy

            count_steps += 1

    #writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
    #writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
    #writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
    #writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
    #writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
    #writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)


if __name__ == "__main__":
    #mkdir('.', 'checkpoints')
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    #args = parser.parse_args()
    #writer = SummaryWriter(comment="ppo_" + args.name)
    render = False
    env = gym.make('Humanoid-v2')
    env.seed(42)
    size = lambda x: int(x.shape[0])
    n_inputs = size(env.observation_space)
    n_outputs = size(env.action_space)

    model = ActorCritic(n_inputs, n_outputs, HIDDEN_SIZE)
    print("model:\n{}".format(model))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=PPO_EPSILON)

    frame_idx  = 0
    train_epoch = 0 #one complte update cycle
    best_reward = None

    state = env.reset() # 8 actions, 8 reward,s 8 dones
    early_stop = False
    #state = tf.convert_to_tensor(state)
    with tf.Session() as sess:
        while not early_stop:

            log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []

            for q in range(PPO_STEPS): #each ppo steps generates actions, states, rewards
                #state = tf.dtypes.cast(state, tf.float32)
                #print(state)
                action, value, norm_dist = model.act(state)
                #print(q)
                next_state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                log_prob_ = norm_dist.log_prob(action)
                log_probs.append(log_prob_)
                values.append(value)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                masks.append(1 - done)

                state = next_state
                frame_idx += 1

            _, next_value, _ = model.act(next_state)
            next_value = next_value[0][0]
            returns = compute_gae(next_value, rewards, masks, values)

            returns  = tf.concat(returns,0)
            #print("returns shape:", returns.shape)

            values = tf.transpose(tf.concat(values,1))
            #print("values shape:", values.shape)

            states = tf.transpose(tf.stack(states,1))
            #print("states shape:", states.shape)

            log_probs  = tf.reshape(tf.concat(log_probs,0), [2,17]) #ppo_size, action_space_size
            #print("log_probs shape:", log_probs.shape)

            actions = tf.transpose(tf.stack(actions,1))
            #print("actions shape:", actions.shape)



            advantage = returns - values
            advantage = normalize(advantage)
            ppo_update(frame_idx, states, actions, log_probs, returns, advantage)

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
PPO_STEPS           = 10 # number of transitions we sample for each training iteration, each step collects a transitoins from each parallel env, hence total amount of data collected = N_envs * PPOsteps = buffer of 2048 data samples to train on
MINI_BATCH_SIZE     = 5 # num of samples that are randomly  selected from the total amount of stored data
PPO_EPOCHS          = 256 #
'''one epoch means one PPO-epochs -- one epochd means one pass over the entire buffer of training data.
So if one buffer has 2048 transitions and mini-batch-size is 64, then one epoch would be 32 selected mini batches.
'''
TEST_EPOCHS         = 10 # how often we run tests to eval our network, one epoch is one entire ppo update cycle
NUM_TESTS           = 1 #num of tests we run to average the total rewards, each time we want eval the performance of the network
TARGET_REWARD       = 400

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def test_env(env, model, deterministic=False):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:

        action, _, norm_dist = model.act(state)
        action = norm_dist.mean()[0] if deterministic \
            else action
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward

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
    batch_size = int(states.get_shape()[0])
    # print(batch_size)
    # print("returns shape:{}\ttype:{}".format(returns.shape, type(returns)))
    # print("values shape:{}\ttype:{}".format(values.shape,type(values)))
    # print("log_probs shape:{}\ttype:{}".format(log_probs.shape,type(log_probs)))
    # print("actions shape:{}\ttype:{}".format(actions.shape,type(actions)))
    # print("states shape:{}\ttype:{}".format(states.shape,type(states)))
    # #print(batch_size//MINI_BATCH_SIZE)
    # # generates random mini-batches until we have covered the full batch

    for _ in range(batch_size // MINI_BATCH_SIZE):
       rand_ids = tf.constant(np.random.randint(0, batch_size, MINI_BATCH_SIZE))
       yield tf.nn.embedding_lookup(states,rand_ids), tf.nn.embedding_lookup(actions,rand_ids), \
             tf.nn.embedding_lookup(log_probs,rand_ids), tf.nn.embedding_lookup(returns,rand_ids), \
             tf.nn.embedding_lookup(advantage,rand_ids)
def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON):

    count_steps, sum_returns, sum_advantage, sum_loss_actor =  0, 0.0, 0.0, 0.0
    sum_loss_critic, sum_entropy, sum_loss_total = 0.0, 0.0, 0.0


    # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
    for _ in range(PPO_EPOCHS):
        # grabs random mini-batches several times until we have covered all data
        #for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            state = sess.run(state)
            _, _, norm_dist = model.act(state,action_=False)
            entropy = norm_dist.entropy()
            new_log_probs = norm_dist.log_prob(action)
            ratio = tf.exp(new_log_probs - old_log_probs)
            actor_loss = -tf.reduce_mean(tf.minimum(
                tf.multiply(ratio, advantage), #surrogate 1
                tf.clip_by_value(ratio, 1.-clip_param, 1.+clip_param)*advantage))

            critic_loss = tf.reduce_mean(tf.square(tf.subtract(return_, value)))
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
    mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    args = parser.parse_args()
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

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        while not early_stop:

            log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []

            for q in range(PPO_STEPS): #each ppo steps generates actions, states, rewards

                action, value, norm_dist = model.act(state)
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
            values = tf.transpose(tf.concat(values,1))
            states = tf.transpose(tf.stack(states,1))
            log_probs  = tf.reshape(tf.concat(log_probs,0), [2,17]) #ppo_size, action_space_size
            actions = tf.transpose(tf.stack(actions,1))
            #print("returns shape:", returns.shape
            #print("values shape:", values.shape)
            #print("log_probs shape:", log_probs.shape)
            #print("actions shape:", actions.shape)
            #print("states shape:", states.shape)



            advantage = returns - values
            advantage = normalize(advantage)
            ppo_update(frame_idx, states, actions, log_probs, returns, advantage)

            if train_epoch % TEST_EPOCHS == 0:
                test_reward = np.mean([test_env(env, model) for _ in range(NUM_TESTS)])
                #writer.add_scalar("test_rewards", test_reward, frame_idx)
                print('Frame %s. reward: %s' % (frame_idx, test_reward))
                # Save a checkpoint every time we achieve a best reward
                if best_reward is None or best_reward < test_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                        name = "%s_best_%+.3f_%d.dat" % (args.name, test_reward, frame_idx)
                        fname = os.path.join('.', 'checkpoints', name)
                        print(fname)
                        #torch.save(model, fname)
                        saver.save(sess, 'my-test-model')
                    best_reward = test_reward
                if test_reward > TARGET_REWARD: early_stop = True

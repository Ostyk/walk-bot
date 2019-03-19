"""
A simple version of Proximal Policy Optimization (PPO)
Based on:
1. [https://arxiv.org/abs/1707.02286]
2. [https://arxiv.org/abs/1707.06347]
View more on this tutorial website: https://morvanzhou.github.io/tutorials
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 600
EP_LEN = 2
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 64
A_UPDATE_STEPS = 20
C_UPDATE_STEPS = 20
S_DIM, A_DIM = 376, 17
METHOD = dict(name='clip', epsilon=0.2)                # Clipped surrogate objective, find this is better
															# epsilon=0.2 is in the paper

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        print(self.tfs)

        # CRITIC #######################################
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, name='layer1-critic')
            self.v = tf.layers.dense(l1, 1, name = 'V_layer')
        # Implementation the Train method
        with tf.variable_scope('ctrain'):
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs) # insted SGD

        # ACTOR ########################################
        # Current policy
        pi, pi_params = self._build_anet('pi', trainable=True)
        with tf.variable_scope('sample_action'):

            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action

        # Hold policy
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('update_oldpi'): # Intercambia los pesos de las capas de hold_pi by pi
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # PPO implementation, Loss function
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate_pp'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        # Implementation the Train method
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        ################################################
        # Almacenando modelo en la carpeta log
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable): # Build the current & hold structure for the policies
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable, name = 'mu_'+name)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable,name ='sigma_'+name )
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma) # Loc is the mean
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) #Recolecta los pesos de los layers l1,mu/2,sigma
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        print(self.sample_op.shape,"THIS IS IT")
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        print(a)
        return np.clip(a, -2, 2) # limita la salida de valores entre -2 & 2, a cada uno de los valores de 'a'

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0] # Salida de NN del Critic|| V = learned state-value function

env = gym.make('Humanoid-v2').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a) # observation, reward, done, info|| 'a' is torque
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        #print(r)
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_) # Obteniendo la respuesta de la NN del Critic, entregando el estado 's_'
            						# V = learned state-value function
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br) # Entranar el Cliente y el actor (Estado, acciones, discounted_r)

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

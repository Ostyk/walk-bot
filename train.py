import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import time

class PPO(object):
    def __init__(self, env):
        #TO DO: turn hyperparmets into a dict given to the class in main.py
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.A_UPDATE_STEPS = 20
        self.C_UPDATE_STEPS = 20
        size = lambda box: int(str(box).split(',')[0].split('(')[1])
        self.S_DIM, self.A_DIM = size(env.observation_space), size(env.action_space)
        METHOD = dict(name='clip', epsilon=0.2)                # Clipped surrogate objective, find this is better
        															# epsilon=0.2 is in the paper
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # CRITIC #######################################
        #A Critic that measures how good the action taken is (value-based)
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, name='layer1-critic')
            self.v = tf.layers.dense(l1, 1, name = 'V_layer')

        # Implementation the Train method -- critic learning method
        with tf.variable_scope('ctrain'):
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs) # instead of SGD

        # ACTOR ########################################
        # an Actor that controls how our agent behaves (policy-based)
        # Current policy
        pi, pi_params = self._build_anet('pi', trainable=True)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1))#, axis=0)       # choosing action

        # Hold policy
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('update_oldpi'): # replace the weights of the hold_pi by pi layers
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # PPO implementation, Loss function
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate_pp'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv # first surrogate function

            self.aloss = -tf.reduce_mean(tf.minimum(
                surr, #second surrogate fiunction
                tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        # Implementation the Train method
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        ################################################
        # Storing model in the log folder
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.A_UPDATE_STEPS)]
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable): # Build the current & hold structure for the policies
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.tanh, trainable=trainable)
            mu = 0.4 * tf.layers.dense(l1, self.A_DIM, tf.nn.tanh, trainable=trainable, name = 'mu_'+name) # shuld be the action space low and high
            sigma = tf.layers.dense(l1, self.A_DIM, tf.nn.softmax, trainable=trainable,name ='sigma_'+name ) #STD
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma) # Loc is the mean
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) #Collect the weights of the layers l1, mu / 2, sigma
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        return a
        #return np.clip(a, -2, 2) # limita la salida de valores entre -2 & 2, a cada uno de los valores de 'a'
        #return np.clip(a, -2, 2)
        #print(a.shape)
        #return np.clip(a, -1, 1) # limits the output of values between -2 and 2, to each of the values of 'a'
        # TO DO: check with humanoid output

    def get_v(self, s):
        #print("DIMENSION", s.ndim)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0] # Exit of NN from Critic, where V = learned state-value function

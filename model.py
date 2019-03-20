
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym


class ActorCritic(object):
    def __init__(self, n_inputs, n_outputs, hidden_size, std=0.0):

        #super(ActorCritic, self).__init__()

        with tf.Session() as self.sess:

            self.n_inputs = tf.placeholder(tf.float32, [None, n_inputs], 'states')
            self.n_outputs = tf.placeholder(tf.float32, [None, n_outputs], 'actions')
            self.hidden_size = hidden_size

            with tf.variable_scope('critic'): #value_net
                layer_1 = tf.layers.dense(self.n_inputs, self.hidden_size, tf.nn.relu, name='layer1-critic')
                self.critic = tf.layers.dense(layer_1, 1, name = 'V_layer')

            with tf.variable_scope('actor'): #Policu_net
                layer1 = tf.layers.dense(self.n_inputs, self.hidden_size, tf.nn.relu, trainable=True, name='layer1-actor')
                self.actor = tf.layers.dense(layer_1, self.hidden_size, name = 'A_layer')
                self.mu = 0.4 * tf.layers.dense(layer1, n_outputs, tf.nn.tanh, trainable=True, name = 'mu_')
                self.sigma = tf.layers.dense(layer1, n_outputs, tf.nn.softplus, trainable=True,name ='sigma_' )
            #self.log_std = tf.multiply(tf.ones(1, self.n_outputs) * std)
            #self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std #seperate parameter not comnnected to the layers of the N, is more efficient
        #tf.summary.FileWriter("log_t/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def act(self, x):
        #x = self.sess.run(x)
        norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
        sample = tf.squeeze(norm_dist.sample(1), axis=0)
        x = x[np.newaxis, :]

        action = self.sess.run(sample, {self.n_inputs: x})[0]
        value = self.sess.run(self.critic, {self.n_inputs: x})
        #log_prob = norm_dist.log_prob(action)
        return action, value, norm_dist

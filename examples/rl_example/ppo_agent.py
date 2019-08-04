import numpy as np
import tensorflow as tf

class PPOAgent():
    MAIN_SCOPE = "main"
    ACTOR_SCOPE = "actor"
    CRITIC_SCOPE = "critic"

    def __init__(self, env, sess):
        self._env = env
        self._sess = sess

        self._build_nets()

        return

    def train(self):
        return

    def get_observation_size(self):
        return self._env.observation_space.shape[-1]

    def get_action_size(self):
        return self._env.action_space.shape[-1]

    def _build_nets(self):
        o_size = self.get_observation_size()
        a_size = self.get_action_size()

        self._o_tf = tf.placeholder(tf.float32, shape=[None, o_size], name="o")
        self._a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self._tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        self._adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
        self._old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")
        
        actor_inputs = [self._o_tf]
        critic_inputs = [self._o_tf]

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.ACTOR_SCOPE):
                self._a_pd_tf = self._build_net_actor(actor_inputs)
            with tf.variable_scope(self.CRITIC_SCOPE):
                self._critic_tf = self._build_net_critic(critic_inputs)

        return

    def _build_net_actor(self, input_tfs):
        input_tf = tf.concat(axis=-1, values=input_tfs)
        return

    def _build_net_critic(self, input_tfs):
        input_tf = tf.concat(axis=-1, values=input_tfs)
        return
import numpy as np
import tensorflow as tf
import rl_path

class PPOAgent():
    MAIN_SCOPE = "main"
    ACTOR_SCOPE = "actor"
    CRITIC_SCOPE = "critic"

    def __init__(self, env, 
                 sess,
                 graph,
                 max_iters=np.inf,
                 epochs=1,
                 batch=1024,#4096,
                 actor_minibatch=256,
                 critic_minibatch=256,
                 actor_layer_sizes=[128, 64],
                 critic_layer_sizes=[128, 64],
                 activation=tf.nn.relu,
                 init_norm_a_std=0.1,
                 discount=0.99,
                 ratio_clip=0.2,
                 actor_stepsize=0.0001,
                 actor_momentum=0.9,
                 critic_stepsize=0.001,
                 critic_momentum=0.9):

        self._env = env
        self._sess = sess
        self._graph = graph
        
        self._epochs = epochs
        self._max_iters = max_iters
        self._batch = batch
        self._actor_minibatch = actor_minibatch
        self._critic_minibatch = critic_minibatch

        self._discount = discount
        self._ratio_clip = ratio_clip

        self._actor_stepsize = actor_stepsize
        self._actor_momentum = actor_momentum
        self._critic_stepsize = critic_stepsize
        self._critic_momentum = critic_momentum

        with self._sess.as_default(), self._graph.as_default():
            self._build_normalizers()
            self._build_nets(actor_layer_sizes, critic_layer_sizes,
                             activation, init_norm_a_std)
            self._build_losses()
            self._build_solvers()

            self._initialize_vars()

        return

    def train(self):
        iter = 0
        while (iter < self._max_iters):
            paths = self._collect_batch()
            update_info = self._update_step(paths)
            self._log_info(update_info)
            iter += 1

        return

    def get_observation_size(self):
        return self._env.observation_space.shape[-1]

    def get_action_size(self):
        return self._env.action_space.shape[-1]

    def sample_actions(self, o):
        single_step = (len(o.shape) == 1)
        if (single_step):
            o = np.expand_dims(o, axis=0)

        feed = {
            self._o_tf: o
        }
        a, logp = self._sess.run([self._sample_a_tf, self._sample_a_logp_tf], feed)

        if (single_step):
            a = a[0]
            logp = logp[0]

        return a, logp

    def _build_normalizers(self):
        if (np.array_equal(self._env.action_space.low, self._env.action_space.high)
            or not np.all(np.isfinite(self._env.action_space.low))
            or not np.all(np.isfinite(self._env.action_space.high))):
            self._a_offset = np.zeros_like(self._env.action_space.low)
            self._a_scale = np.ones_like(self._env.action_space.low)
        else:
            self._a_offset = -0.5 * (self._env.action_space.high + self._env.action_space.low)
            self._a_scale = 2 / (self._env.action_space.high - self._env.action_space.low)

            
        if (np.array_equal(self._env.observation_space.low, self._env.observation_space.high)
            or not np.all(np.isfinite(self._env.observation_space.low))
            or not np.all(np.isfinite(self._env.observation_space.high))):
            self._o_offset = np.zeros_like(self._env.observation_space.low)
            self._o_scale = np.ones_like(self._env.observation_space.low)
        else:
            self._o_offset = -0.5 * (self._env.observation_space.high + self._env.observation_space.low)
            self._o_scale = 2 / (self._env.observation_space.high - self._env.observation_space.low)

        return

    def _build_nets(self, actor_layer_sizes, critic_layer_sizes,
                    activation, init_norm_a_std):
        o_size = self.get_observation_size()
        a_size = self.get_action_size()

        self._o_tf = tf.placeholder(tf.float32, shape=[None, o_size], name="o")
        self._a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self._tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        self._adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
        self._old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")

        norm_o_tf = self._normalize_o(self._o_tf)
        norm_a_tf = self._normalize_a(self._a_tf)
        
        actor_inputs = [norm_o_tf]
        critic_inputs = [norm_o_tf]

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.ACTOR_SCOPE):
                self._actor_pd_tf = self._build_net_actor(actor_inputs, a_size,
                                                          actor_layer_sizes, activation,
                                                          init_norm_a_std, reuse=False)
                sample_norm_a_tf = self._actor_pd_tf.sample()
                self._sample_a_tf = self._unnormalize_a(sample_norm_a_tf)
                self._sample_a_logp_tf = self._actor_pd_tf.log_prob(sample_norm_a_tf)
                self._a_logp_tf = self._actor_pd_tf.log_prob(norm_a_tf)

            with tf.variable_scope(self.CRITIC_SCOPE):
                self._critic_tf = self._build_net_critic(critic_inputs, critic_layer_sizes,
                                                         activation, reuse=False)

        return

    def _build_losses(self):
        val_diff = self._tar_val_tf - self._critic_tf
        self._critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(val_diff))

        ratio_tf = tf.exp(self._a_logp_tf - self._old_logp_tf)
        actor_loss0 = self._adv_tf * ratio_tf
        actor_loss1 = self._adv_tf * tf.clip_by_value(ratio_tf, 1.0 - self._ratio_clip, 1 + self._ratio_clip)
        self._actor_loss_tf = -tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))
        
        self._clip_frac_tf = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self._ratio_clip)))

        return

    def _build_solvers(self):
        critic_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.CRITIC_SCOPE)
        self._critic_opt = tf.train.MomentumOptimizer(learning_rate=self._critic_stepsize, momentum=self._critic_momentum)
        self._update_critic_op = self._critic_opt.minimize(self._critic_loss_tf, var_list=critic_vars)

        actor_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.ACTOR_SCOPE)
        self._actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize, momentum=self._actor_momentum)
        self._update_actor_op = self._actor_opt.minimize(self._actor_loss_tf, var_list=actor_vars)

        return

    def _tf_vars(self, scope=''):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        return vars

    def _build_net_actor(self, input_tfs, a_size, layer_sizes, activation, init_std, reuse=False):
        h_weight_init = tf.contrib.layers.xavier_initializer()
        mean_weight_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        mean_bias_init = tf.zeros_initializer()
        logstd_weight_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        logstd_bias_init = tf.constant_initializer(np.log(init_std))

        input_tf = tf.concat(axis=-1, values=input_tfs)
        curr_tf = input_tf
        for i, size in enumerate(layer_sizes):
            with tf.variable_scope(str(i), reuse=reuse):
                curr_tf = tf.layers.dense(inputs=curr_tf, units=size,
                                        kernel_initializer=h_weight_init,
                                        activation=activation)

        mean_tf = tf.layers.dense(inputs=curr_tf, units=a_size,
                                        kernel_initializer=mean_weight_init,
                                        bias_initializer=mean_bias_init,
                                        activation=None)
        logstd_tf = tf.layers.dense(inputs=curr_tf, units=a_size,
                                        kernel_initializer=logstd_weight_init,
                                        bias_initializer=logstd_bias_init,
                                        activation=None)
        std_tf = tf.exp(logstd_tf)

        a_pd_tf = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_tf, scale_diag=std_tf)
   
        return a_pd_tf

    def _build_net_critic(self, input_tfs, layer_sizes, activation, reuse=False):
        h_weight_init = tf.contrib.layers.xavier_initializer()
        output_weight_init = tf.contrib.layers.xavier_initializer()
        output_bias_init = tf.zeros_initializer()

        input_tf = tf.concat(axis=-1, values=input_tfs)
        curr_tf = input_tf
        for i, size in enumerate(layer_sizes):
            with tf.variable_scope(str(i), reuse=reuse):
                curr_tf = tf.layers.dense(inputs=curr_tf, units=size,
                                        kernel_initializer=h_weight_init,
                                        activation=activation)

        output_tf = tf.layers.dense(inputs=curr_tf, units=1,
                                    kernel_initializer=output_weight_init,
                                    bias_initializer=output_bias_init,
                                    activation=None)
        output_tf = tf.squeeze(output_tf)

        return output_tf

    def _initialize_vars(self):
        self._sess.run(tf.global_variables_initializer())
        return
    
    def _normalize_o(self, o):
        norm_o = (o + self._o_offset) * self._o_scale
        return norm_o

    def _unnormalize_o(self, norm_o):
        o = norm_o / self._o_scale - self._o_offset
        return o

    def _normalize_a(self, a):
        norm_a = (a + self._a_offset) * self._a_scale
        return norm_a

    def _unnormalize_a(self, norm_a):
        a = norm_a / self._a_scale - self._a_offset
        return a

    def _collect_batch(self):
        sample_count = 0
        paths = []

        while sample_count < self._batch:
            curr_path = self._rollout_path()
            paths.append(curr_path)

            path_len = curr_path.pathlength()
            sample_count += path_len

        return paths

    def _rollout_path(self):
        curr_path = rl_path.RLPath()

        o = self._env.reset()
        o = np.array(o)
        curr_path.observations.append(o)

        done = False
        while not done:
            a, logp = self.sample_actions(o)
            o, r, done, info = self._env.step(a)
            o = np.array(o)
                
            curr_path.observations.append(o)
            curr_path.actions.append(a)
            curr_path.rewards.append(r)
            curr_path.logps.append(logp)

        return curr_path

    def _update_step(self, paths):
        observations, actions, logps, rewards, last_step = self._flatten_paths(paths)
        n = len(observations)

        returns, adv = self._calc_returns_and_advantages(observations, rewards, last_step)

        idx = np.array(list(range(n)))
        for i in range(self._epochs):
            np.random.shuffle(idx)

            critic_minibatches = n // self._critic_minibatch
            actor_minibatches = n // self._actor_minibatch

        return

    def _log_info(self, info):

        return

    def _flatten_paths(self, paths):
        dummy_action = np.zeros(self.get_action_size())
        dummy_logp = 0.0
        dummy_reward = 0.0

        observations = np.concatenate([p.observations for p in paths])
        actions = np.concatenate([p.actions + [dummy_action] for p in paths])
        logps = np.concatenate([p.logps + [dummy_logp] for p in paths])
        rewards = np.concatenate([p.rewards + [dummy_reward] for p in paths])
        last_step = np.concatenate([[0.0] * p.pathlength() + [1.0] for p in paths])

        return observations, actions, logps, rewards, last_step

    def _calc_returns_and_advantages(self, observations, rewards, last_step):
        # hack
        returns = np.zeros_like(rewards)
        adv = np.zeros_like(adv)
        return returns, adv
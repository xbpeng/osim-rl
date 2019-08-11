import os
import time
import numpy as np
import tensorflow as tf
import rl_path
import tf_normalizer
import logger

class PPOAgent():
    MAIN_SCOPE = "main"
    ACTOR_SCOPE = "actor"
    CRITIC_SCOPE = "critic"
    RESOURCE_SCOPE = "resource"

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
                 td_lambda=0.95,
                 ratio_clip=0.2,
                 actor_stepsize=0.0001,
                 actor_momentum=0.9,
                 critic_stepsize=0.001,
                 critic_momentum=0.9,
                 output_dir="output/"):

        self._env = env
        self._sess = sess
        self._graph = graph
        
        self._epochs = epochs
        self._max_iters = max_iters
        self._batch = batch
        self._actor_minibatch = actor_minibatch
        self._critic_minibatch = critic_minibatch

        self._discount = discount
        self._td_lambda = td_lambda
        self._ratio_clip = ratio_clip

        self._actor_stepsize = actor_stepsize
        self._actor_momentum = actor_momentum
        self._critic_stepsize = critic_stepsize
        self._critic_momentum = critic_momentum

        self._output_dir = output_dir
        
        self._logger = self._build_logger()

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
        samples = 0
        start_time = time.time()

        while (iter < self._max_iters):
            paths = self._collect_batch()
            update_info = self._update(paths)

            curr_samples = sum([p.pathlength() for p in paths])
            samples += curr_samples
            
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours

            train_return = np.mean([p.calc_return() for p in paths])
            path_count = len(paths)

            update_info["iter"] = iter
            update_info["wall_time"] = wall_time
            update_info["samples"] = samples
            update_info["train_return"] = train_return
            update_info["train_path_count"] = path_count
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

    def eval_critic(self, observations):
        feed = {
            self._o_tf: observations
        }
        vals = self._sess.run(self._critic_tf, feed)
        return vals

    def _build_normalizers(self):
        if (np.array_equal(self._env.action_space.low, self._env.action_space.high)
            or not np.all(np.isfinite(self._env.action_space.low))
            or not np.all(np.isfinite(self._env.action_space.high))):
            a_mean = np.zeros_like(self._env.action_space.low)
            a_std = np.ones_like(self._env.action_space.low)
        else:
            a_mean = 0.5 * (self._env.action_space.high + self._env.action_space.low)
            a_std = 0.5 * (self._env.action_space.high - self._env.action_space.low)

            
        if (np.array_equal(self._env.observation_space.low, self._env.observation_space.high)
            or not np.all(np.isfinite(self._env.observation_space.low))
            or not np.all(np.isfinite(self._env.observation_space.high))):
            o_mean = np.zeros_like(self._env.observation_space.low)
            o_std = np.ones_like(self._env.observation_space.low)
        else:
            o_mean = 0.5 * (self._env.observation_space.high + self._env.observation_space.low)
            o_std = 0.5 * (self._env.observation_space.high - self._env.observation_space.low)
            
        with tf.variable_scope(self.RESOURCE_SCOPE):
            self._a_norm = tf_normalizer.TFNormalizer(self._sess, "a_norm", self.get_action_size(),
                                                init_mean=a_mean, init_std=a_std)
            self._o_norm = tf_normalizer.TFNormalizer(self._sess, "o_norm", self.get_observation_size(),
                                                init_mean=o_mean, init_std=o_std)
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

        norm_o_tf = self._o_norm.normalize_tf(self._o_tf)
        norm_a_tf = self._a_norm.normalize_tf(self._a_tf)
        
        actor_inputs = [norm_o_tf]
        critic_inputs = [norm_o_tf]

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.ACTOR_SCOPE):
                self._actor_pd_tf = self._build_net_actor(actor_inputs, a_size,
                                                          actor_layer_sizes, activation,
                                                          init_norm_a_std, reuse=False)
                sample_norm_a_tf = self._actor_pd_tf.sample()
                self._sample_a_tf = self._a_norm.unnormalize_tf(sample_norm_a_tf)
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

    def _build_logger(self):
        output_file = os.path.join(self._output_dir, "log.txt")
        log = logger.Logger()
        log.configure_output_file(output_file)
        return log
    
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

    def _update(self, paths):
        observations, actions, logps, rewards, last_step = self._flatten_paths(paths)
        n = len(observations)

        tar_vals, advs = self._calc_values_and_advantages(observations, rewards, last_step)
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-5)
        advs = np.clip(advs, -4.0, 4.0)

        idx = np.array(list(range(n)))
        not_last_step = np.logical_not(last_step)
        valid_idx = idx[not_last_step]

        info = None
        for i in range(self._epochs):
            np.random.shuffle(valid_idx)
            critic_info = self._update_critic(observations, tar_vals, valid_idx)
            actor_info = self._update_actor(observations, actions, logps, advs, valid_idx)

            curr_info = dict({**critic_info, **actor_info})

            if (info is None):
                info = curr_info
            else:
                for key in info.keys():
                    info[key] += curr_info[key]

        for key in info.keys():
            info[key] /= self._epochs

        self._update_normalizers(paths)

        return info

    def _log_info(self, info):
        self._logger.log_tabular("iter", int(info["iter"]))
        self._logger.log_tabular("wall_time", info["wall_time"])
        self._logger.log_tabular("samples", int(info["samples"]))
        self._logger.log_tabular("critic_loss", info["critic_loss"])
        self._logger.log_tabular("actor_loss", info["actor_loss"])
        self._logger.log_tabular("actor_clip_frac", info["actor_clip_frac"])
        self._logger.log_tabular("train_return", info["train_return"])
        self._logger.log_tabular("train_path_count", int(info["train_path_count"]))

        self._logger.print_tabular()
        self._logger.dump_tabular()
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

    def _calc_values_and_advantages(self, observations, rewards, last_step):
        pred_vals = self.eval_critic(observations)

        next_vals = pred_vals[1:]
        not_last_step = np.logical_not(last_step)

        discounts = self._discount * not_last_step[1:]
        delta = rewards[:-1] + discounts * next_vals - pred_vals[:-1]
        weighted_discounts = self._td_lambda * discounts

        advs = np.zeros_like(delta)
        accum_td = 0.0
        for i in reversed(range(len(advs))):
            curr_weighted_discount = weighted_discounts[i]
            curr_delta = delta[i]
            curr_adv = curr_delta + curr_weighted_discount * accum_td
            accum_td = curr_adv
            advs[i] = curr_adv

        vals = pred_vals[:-1] + advs
        
        return vals, advs

    def _update_critic(self, observations, tar_vals, valid_idx):
        num_valid = len(valid_idx)
        num_batches = num_valid // self._critic_minibatch

        total_loss = 0.0
        for b in range(num_batches):
            curr_batch = valid_idx[(b * self._critic_minibatch):((b + 1) * self._critic_minibatch)]
            curr_observations = observations[curr_batch]
            curr_tar_vals = tar_vals[curr_batch]

            feed = {
                self._o_tf: curr_observations,
                self._tar_val_tf: curr_tar_vals
            }
            _, curr_loss = self._sess.run([self._update_critic_op, self._critic_loss_tf], feed)

            total_loss += curr_loss

        avg_loss = total_loss / num_batches
        info = dict({
            "critic_loss":avg_loss
        })

        return info

    def _update_actor(self, observations, actions, logps, advs, valid_idx):
        num_valid = len(valid_idx)
        num_batches = num_valid // self._actor_minibatch

        total_loss = 0.0
        total_clip_frac = 0.0
        for b in range(num_batches):
            curr_batch = valid_idx[(b * self._actor_minibatch):((b + 1) * self._actor_minibatch)]
            curr_observations = observations[curr_batch]
            curr_actions = actions[curr_batch]
            curr_logps = logps[curr_batch]
            curr_advs = advs[curr_batch]

            feed = {
                self._o_tf: curr_observations,
                self._a_tf: curr_actions,
                self._old_logp_tf: curr_logps,
                self._adv_tf: curr_advs
            }
            _, curr_loss, clip_frac = self._sess.run([self._update_actor_op, self._actor_loss_tf,
                                                      self._clip_frac_tf], feed)
            total_loss += curr_loss
            total_clip_frac += clip_frac

        avg_loss = total_loss / num_batches
        avg_clip_frac = total_clip_frac / num_batches
        info = dict({
            "actor_loss": avg_loss,
            "actor_clip_frac": avg_clip_frac
        })

        return info

    def _update_normalizers(self, paths):
        for p in paths:
            self._o_norm.record(np.array(p.observations))
        self._o_norm.update()
        return
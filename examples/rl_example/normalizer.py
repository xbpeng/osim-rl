import numpy as np
import copy

class Normalizer(object):
    def __init__(self, size, init_mean=None, init_std=None, eps=0.02, clip=np.inf):
        self.eps = eps
        self.clip = clip
        self.mean = np.zeros(size)
        self.std = np.ones(size)
        self.count = 0

        if init_mean is not None:
            if not isinstance(init_mean, np.ndarray):
                assert(size == 1)
                init_mean = np.array([init_mean])

            assert init_mean.size == size, \
            print('Normalizer init mean shape mismatch, expecting size {:d}, but got {:d}'.format(size, init_mean.size))
            self.mean = init_mean

        if init_std is not None:
            if not isinstance(init_std, np.ndarray):
                assert(size == 1)
                init_std = np.array([init_std])

            assert init_std.size == size, \
            print('Normalizer init std shape mismatch, expecting size {:d}, but got {:d}'.format(size, init_std.size))
            self.std = init_std

        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        
        self._new_count = 0
        self._new_sum = np.zeros_like(self.mean)
        self._new_sum_sq = np.zeros_like(self.mean_sq)
        
        return

    def record(self, x):
        size = self.get_size()
        is_array = isinstance(x, np.ndarray)
        if not is_array:
            assert(size == 1)
            x = np.array([[x]])

        assert x.shape[-1] == size, \
            print('Normalizer shape mismatch, expecting size {:d}, but got {:d}'.format(size, x.shape[-1]))
        x = np.reshape(x, [-1, size])

        self._new_count += x.shape[0]
        self._new_sum += np.sum(x, axis=0)
        self._new_sum_sq += np.sum(np.square(x), axis=0)
        return

    def update(self):
        if self._new_count > 0:
            new_total = self.count + self._new_count

            new_mean = self._new_sum / self._new_count
            new_mean_sq = self._new_sum_sq / self._new_count
            w_old = float(self.count) / new_total
            w_new = float(self._new_count) / new_total

            self.mean = w_old * self.mean + w_new * new_mean
            self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
            self.count = new_total
            self.std = self.calc_std(self.mean, self.mean_sq)

            self._new_count = 0
            self._new_sum.fill(0)
            self._new_sum_sq.fill(0)

        return

    def get_size(self):
        return self.mean.size

    def set_mean_std(self, mean, std):
        size = self.get_size()
        is_array = isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)
        
        if not is_array:
            assert(size == 1)
            mean = np.array([mean])
            std = np.array([std])

        assert len(mean) == size and len(std) == size, \
            print('Normalizer shape mismatch, expecting size {:d}, but got {:d} and {:d}'.format(size, len(mean), len(std)))
        
        self.mean = mean
        self.std = std
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def normalize(self, x):
        norm_x = (x - self.mean) / self.std
        norm_x = np.clip(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize(self, norm_x):
        x = norm_x * self.std + self.mean
        return x

    def calc_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        # some time floating point errors can lead to small negative numbers
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    def calc_mean_sq(self, mean, std):
        return np.square(std) + np.square(self.mean)
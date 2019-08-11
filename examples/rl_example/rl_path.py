import numpy as np
import tensorflow as tf

class RLPath():
    def __init__(self):
        self.observations = []
        self.actions = []
        self.logps = []
        self.rewards = []

        self.clear()
        return

    def pathlength(self):
        return len(self.actions)

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()
        return

    def calc_return(self):
        return sum(self.rewards)
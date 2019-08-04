import sys
import argparse
import tensorflow as tf

from osim.env import arm
import ppo_agent

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test neural net motor controller")
    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--test", dest="train", action="store_false", default=True)
    parser.add_argument("--steps", dest="steps", action="store", default=10000, type=int)
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--model", dest="model", action="store", default="example.h5f")
    args = parser.parse_args()
    return args

def build_env():
    env = arm.Arm2DEnv(args.visualize)
    return env

def build_agent(env, sess):
    agent = ppo_agent.PPOAgent(env, sess)
    return agent

def main():
    global args
    args = parse_args()

    env = build_env()
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = build_agent(env, sess)

    agent.train()

    return

if __name__ == "__main__":
    main()
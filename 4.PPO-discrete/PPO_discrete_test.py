import argparse
import datetime
import logging

import gym
import numpy as np
import torch

from normalization import Normalization, RewardScaling
from ppo_discrete import PPO_discrete

logging.getLogger().setLevel(logging.INFO)


def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s, device=torch.device('cpu'))  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            env.render()
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    max_reward = float('-inf')
    time_now = datetime.datetime.now()
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    agent = PPO_discrete(args)

    agent.actor.load_state_dict(
        torch.load('saved_models/agent-2023-01-23 01:59:27.212059.pth', map_location=torch.device('cpu')))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for i in range(1000):
        evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
        print(f'Reward:{evaluate_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['CartPole-v1', 'LunarLander-v2']
    env_index = 1
    main(args, env_name=env_name[env_index], number=1, seed=0)

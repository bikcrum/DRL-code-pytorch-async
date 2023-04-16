import argparse
import collections
import logging

import gym
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from normalization import Normalization, RewardScaling
from ppo_continuous_transformer import PPO_continuous_Transformer
from replaybuffer import ReplayBuffer


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Create env
        self.env = gym.make(env_name)
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.shape[0]
        args.max_action = float(self.env.action_space.high[0])
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("episode_limit={}".format(args.episode_limit))

        self.replay_buffer = ReplayBuffer(args)
        self.agent = PPO_continuous_Transformer(args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='runs/PPO_continuous/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        device_collector, device_optim = torch.device('cpu'), torch.device('cuda')
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            # logging.info('Evaluating')
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                logging.info('Evaluating')
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            logging.info('Collecting experience')
            _, episode_steps = self.run_episode()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                logging.info('Training')
                actor_loss, critic_loss = self.agent.train(self.replay_buffer, self.total_steps,
                                                           device=device_optim)  # Training
                print("total_steps:{} \t actor_loss:{} \t critic_loss:{}".format(self.total_steps, actor_loss,
                                                                                 critic_loss))
                self.replay_buffer.reset_buffer()
                self.agent.actor = self.agent.actor.to(device_collector)
                self.agent.critic = self.agent.critic.to(device_collector)

        self.evaluate_policy()
        self.env.close()

    def run_episode(self, ):
        episode_reward = 0
        s = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        state_buffer = collections.deque(maxlen=self.args.episode_limit)

        # self.agent.reset_rnn_hidden()
        for episode_step in tqdm.tqdm(range(self.args.episode_limit)):
            if self.args.use_state_norm:
                s = self.state_norm(s)

            if len(state_buffer) == self.args.episode_limit:
                state_buffer.popleft()

            state_buffer.append(s)

            a, a_logprob = self.agent.choose_action(state_buffer, evaluate=False)
            v = self.agent.get_value(state_buffer)

            # Range of a is [-1, 1], so we need to multiply max_action to convert it to the real action range
            s_, r, done, _ = self.env.step(a.flatten().detach().numpy() * self.args.max_action)
            episode_reward += r

            if done and episode_step + 1 != self.args.episode_limit:
                dw = True
            else:
                dw = False
            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)
            # Store the transition
            self.replay_buffer.store_transition(episode_step, s, v, a, a_logprob, r, dw)
            s = s_
            if done:
                break

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            s = self.state_norm(s)

        if len(state_buffer) == self.args.episode_limit:
            state_buffer.popleft()
        state_buffer.append(s)

        v = self.agent.get_value(state_buffer)
        self.replay_buffer.store_last_value(episode_step + 1, v)

        return episode_reward, episode_step + 1

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, done = 0, False
            s = self.env.reset()
            # self.agent.reset_rnn_hidden()

            state_buffer = collections.deque(maxlen=self.args.episode_limit)

            while not done:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                if len(state_buffer) == self.args.episode_limit:
                    state_buffer.popleft()
                state_buffer.append(s)
                a, a_logprob = self.agent.choose_action(state_buffer, evaluate=True)

                # Range of a is [-1, 1], so we need to multiply max_action to convert it to the real action range
                s_, r, done, _ = self.env.step(a.flatten().detach().numpy() * self.args.max_action)
                # self.env.render()
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
                               global_step=self.total_steps)
        # Save the rewards and models
        np.save('./data_train/PPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
                np.array(self.evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-continuous-Transformer")
    parser.add_argument("--max_train_steps", type=int, default=int(3e8), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument('--transformer_max_len', type=int, default=64, help='max length of transformer')
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")

    args = parser.parse_args()

    env_names = ['Pendulum-v1', 'BipedalWalker-v3']
    env_index = 1
    for seed in [0, 10, 100]:
        runner = Runner(args, env_name=env_names[env_index], number=3, seed=seed)
        runner.run()
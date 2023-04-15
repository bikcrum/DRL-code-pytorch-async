import datetime
import logging

import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete_rnn import PPO_discrete_RNN

logging.getLogger().setLevel(logging.INFO)


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
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("episode_limit={}".format(args.episode_limit))

        self.replay_buffer = ReplayBuffer(args)
        self.agent = PPO_discrete_RNN(args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        time_now = datetime.datetime.now()

        wandb.init(
            entity='team-osu',
            project=f'toy-test-{self.env_name}',
            name=str(time_now),
            config=args.__dict__
        )

        device_collector, device_optim = torch.device('cpu'), torch.device('cuda')
        prev_total_steps = 0

        while self.total_steps < self.args.max_train_steps:
            # if self.total_steps // self.args.evaluate_freq > evaluate_num:
            if self.total_steps - prev_total_steps > self.args.evaluate_freq:
                ep_reward, ep_len = self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                # self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                prev_total_steps = self.total_steps

                log = {
                    "episode_reward": ep_reward,
                    "episode_length": ep_len,
                    "total_steps": self.total_steps,
                    "time_elapsed": (datetime.datetime.now() - time_now).total_seconds()
                }
                logging.info(log)

                wandb.log(log, step=self.total_steps)

            ep_reward, ep_len = self.run_episode()  # Run an episode
            self.total_steps += ep_len

            log = {
                "episode_reward_eval": ep_reward / self.args.batch_size,
                "episode_length_eval": ep_len / self.args.batch_size,
                "total_steps": self.total_steps,
                "time_elapsed": (datetime.datetime.now() - time_now).total_seconds()
            }

            logging.info(log)

            wandb.log(log, step=self.total_steps)

            actor_loss, critic_loss = self.agent.train(self.replay_buffer, self.total_steps,
                                    device_optim)  # Training

            self.agent.actor = self.agent.actor.to(device_collector)
            self.agent.critic = self.agent.critic.to(device_collector)

            log = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_steps": self.total_steps,
                "time_elapsed": (datetime.datetime.now() - time_now).total_seconds()
            }
            logging.info(log)

            wandb.log(log, step=self.total_steps)

            self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def run_episode(self, ):
        total_reward = 0
        total_length = 0
        for _ in range(self.args.batch_size):
            episode_reward = 0
            episode_length = 0
            s = self.env.reset()
            if self.args.use_reward_scaling:
                self.reward_scaling.reset()
            self.agent.reset_rnn_hidden()
            for episode_step in range(self.args.episode_limit):
                if self.args.use_state_norm:
                    s = self.state_norm(s)
                a, a_logprob = self.agent.choose_action(s, evaluate=False)
                v = self.agent.get_value(s)
                s_, r, done, _ = self.env.step(a)
                episode_reward += r
                episode_length += 1

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
            v = self.agent.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v)

            total_reward += episode_reward
            total_length += episode_length

        return total_reward, total_length

    def evaluate_policy(self, ):
        evaluate_reward = 0
        evaluate_length = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, episode_length, done = 0, 0, False
            s = self.env.reset()
            self.agent.reset_rnn_hidden()
            while not done:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                a, a_logprob = self.agent.choose_action(s, evaluate=True)
                s_, r, done, _ = self.env.step(a)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            evaluate_length += episode_length

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_length = evaluate_length / self.args.evaluate_times

        # self.evaluate_rewards.append(evaluate_reward)
        # print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        # self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
        #                        global_step=self.total_steps)
        # # Save the rewards and models
        # np.save('./data_train/PPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
        #         np.array(self.evaluate_rewards))

        return evaluate_reward, evaluate_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
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

    env_names = ['CartPole-v1', 'LunarLander-v2']
    env_index = 0
    for seed in [0, 10, 100]:
        runner = Runner(args, env_name=env_names[env_index], number=3, seed=seed)
        runner.run()

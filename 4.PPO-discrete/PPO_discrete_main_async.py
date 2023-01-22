import datetime
import logging
import time

import torch
import numpy as np
import tqdm
import wandb
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
import ray
from copy import deepcopy

logging.getLogger().setLevel(logging.INFO)


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


@ray.remote
def collector(env, state_norm, reward_scaling, agent, reward_norm, batch_size, args):
    curr_buf_size = 0
    replay_buffer = ReplayBuffer(args)

    rewards = []
    lengths = []

    while curr_buf_size < batch_size:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        episode_reward = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            episode_reward += r

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            curr_buf_size += 1

            # if curr_buf_size > batch_size:
                # if len(rewards) == 0:
                #     return replay_buffer, episode_reward, episode_steps
                # else:
                # return replay_buffer, np.mean(rewards), np.mean(lengths)

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_

        rewards.append(episode_reward)
        lengths.append(episode_steps)

    return replay_buffer, np.mean(rewards), np.mean(lengths)


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

    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    run = wandb.init(
        entity='team-osu',
        project=f'toy-test-{env_name}',
        name=str(time_now),
        config=args.__dict__
    )
    # Build a tensorboard
    # writer = SummaryWriter(
    #     log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}_time_{}'.format(env_name, number, seed, str(time_now)))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    reward_scaling = None
    reward_norm = None
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    pbar = tqdm.tqdm(total=args.max_train_steps)

    n_workers = args.n_workers
    _args = deepcopy(args)
    _args.batch_size //= n_workers

    while total_steps < args.max_train_steps:

        # s = env.reset()
        # if args.use_state_norm:
        #     s = state_norm(s)
        # if args.use_reward_scaling:
        #     reward_scaling.reset()
        # episode_steps = 0
        # done = False
        # while not done:
        #     episode_steps += 1
        #     a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
        #     s_, r, done, _ = env.step(a)
        #
        #     if args.use_state_norm:
        #         s_ = state_norm(s_)
        #     if args.use_reward_norm:
        #         r = reward_norm(r)
        #     elif args.use_reward_scaling:
        #         r = reward_scaling(r)
        #
        #     # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
        #     # dw means dead or win,there is no next state s';
        #     # but when reaching the max_episode_steps,there is a next state s' actually.
        #     if done and episode_steps != args.max_episode_steps:
        #         dw = True
        #     else:
        #         dw = False
        #
        #     replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
        #     s = s_
        #     total_steps += 1
        #     pbar.update(1)

        # When the number of transitions in buffer reaches batch_size,then update

        # replay_buffer_1 = collector(env, state_norm, reward_scaling, agent, reward_norm, _args.batch_size, _args)
        # replay_buffer_2 = collector(env, state_norm, reward_scaling, agent, reward_norm, _args.batch_size, _args)
        replay_buffers = ray.get(
            [collector.remote(env, state_norm, reward_scaling, agent, reward_norm, _args.batch_size, _args) for _
             in range(n_workers)])

        # replay_buffer.s = np.vstack([rf.s for rf, _, _ in replay_buffers])
        # replay_buffer.r = np.vstack([rf.r for rf, _, _ in replay_buffers])
        # replay_buffer.a = np.vstack([rf.a for rf, _, _ in replay_buffers])
        # replay_buffer.done = np.vstack([rf.done for rf, _, _ in replay_buffers])
        # replay_buffer.a_logprob = np.vstack([rf.a_logprob for rf, _, _ in replay_buffers])
        # replay_buffer.dw = np.vstack([rf.dw for rf, _, _ in replay_buffers])
        # replay_buffer.s_ = np.vstack([rf.s_ for rf, _, _ in replay_buffers])
        # replay_buffer.count = np.sum([rf.count for rf, _, _ in replay_buffers])

        replay_buffer.s = np.vstack([rf.s for rf, _, _ in replay_buffers])
        replay_buffer.r = np.hstack([rf.r for rf, _, _ in replay_buffers])[:,np.newaxis]
        replay_buffer.a = np.hstack([rf.a for rf, _, _ in replay_buffers])[:,np.newaxis]
        replay_buffer.done = np.hstack([rf.done for rf, _, _ in replay_buffers])[:,np.newaxis]
        replay_buffer.a_logprob = np.hstack([rf.a_logprob for rf, _, _ in replay_buffers])[:,np.newaxis]
        replay_buffer.dw = np.hstack([rf.dw for rf, _, _ in replay_buffers])[:,np.newaxis]
        replay_buffer.s_ = np.vstack([rf.s_ for rf, _, _ in replay_buffers])
        replay_buffer.count = np.sum([len(rf.a) for rf, _, _ in replay_buffers])

        reward = np.mean([r for _, r, _ in replay_buffers])
        length = np.mean([l for _, _, l in replay_buffers])

        # assert replay_buffer.count == args.batch_size

        total_steps += replay_buffer.count
        pbar.update(replay_buffer.count)

        agent.update(replay_buffer, total_steps)

        replay_buffer.count = 0

        # reward = replay_buffer.r[:replay_buffer.count].mean()

        log = {'episode_reward': reward,
               'episode_length': length,
               'total_steps': total_steps,
               'time_elapsed': (datetime.datetime.now() - time_now).seconds}

        logging.info(log)
        wandb.log(log)

        if reward > max_reward:
            max_reward = reward
            torch.save(agent.actor.state_dict(), f'agent-{time_now}.pth')

    run.finish()

    # Save the rewards
    # if evaluate_num % args.save_freq == 0:
    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed),
    #             np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=4096,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of collectors")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
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
    env_index = 0
    main(args, env_name=env_name[env_index], number=1, seed=0)

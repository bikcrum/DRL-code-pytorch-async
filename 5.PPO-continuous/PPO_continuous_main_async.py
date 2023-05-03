import argparse
import datetime
import logging
import os
from copy import deepcopy

import gym
import numpy as np
import ray
import torch
import tqdm
import wandb

from normalization import Normalization, RewardScaling
from ppo_continuous import PPO_continuous
from replaybuffer import ReplayBuffer

logging.getLogger().setLevel(logging.INFO)

ray.init(num_cpus=20)


def evaluate_policy(args, env, agent, state_norm, device):
    n_epoch = 3
    reward = 0
    length = 0
    for _ in range(n_epoch):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_length = 0
        episode_reward = 0
        while not done:
            a = agent.evaluate(s, device)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            # env.render()
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            episode_length += 1
            s = s_
        reward += episode_reward
        length += episode_length

    return reward / n_epoch, length / n_epoch


@ray.remote
def collector(env, state_norm, reward_scaling, actor, reward_norm, batch_size, args, device):
    actor = actor.to(device)

    def choose_action(s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float, device=device), 0)
        if args.policy_dist == "Beta":
            with torch.no_grad():
                dist = actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -args.max_action, args.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    curr_buf_size = 0
    replay_buffer = ReplayBuffer(args)

    while curr_buf_size < batch_size:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            # env.render()

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            curr_buf_size += 1

            if curr_buf_size > args.batch_size:
                break

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_

    return replay_buffer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cuda")
    else:
        try:
            # For apple silicon
            if torch.backends.mps.is_available():
                # May not require in future pytorch after fix
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
                return torch.device("cpu"), torch.device("mps")
            else:
                return torch.device("cpu"), torch.device("cpu")
        except Exception as e:
            logging.error(e)
            return torch.device("cpu"), torch.device("cpu")


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
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    dev_inf, dev_optim = get_device()
    # dev_inf, dev_optim = torch.device('cpu'), torch.device('cpu')

    run = wandb.init(
        entity='team-osu',
        project=f'toy-test-{env_name}',
        name=str(time_now),
        config=args.__dict__
    )

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    reward_norm = None
    reward_scaling = None
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    pbar = tqdm.tqdm(total=args.max_train_steps)

    n_workers = args.n_workers
    _args = deepcopy(args)
    _args.batch_size //= n_workers

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    prev_total_steps = 0

    while total_steps < args.max_train_steps:
        actor = agent.actor.to(dev_inf)

        logging.info("Collecting data")

        replay_buffers = ray.get([collector.remote(env, state_norm, reward_scaling, actor, reward_norm,
                                                   _args.batch_size,
                                                   _args,
                                                   dev_inf) for _ in range(n_workers)])

        replay_buffer.s = np.vstack([rf.s for rf in replay_buffers])
        replay_buffer.r = np.vstack([rf.r for rf in replay_buffers])
        replay_buffer.a = np.vstack([rf.a for rf in replay_buffers])
        replay_buffer.done = np.vstack([rf.done for rf in replay_buffers])
        replay_buffer.a_logprob = np.vstack([rf.a_logprob for rf in replay_buffers])
        replay_buffer.dw = np.vstack([rf.dw for rf in replay_buffers])
        replay_buffer.s_ = np.vstack([rf.s_ for rf in replay_buffers])
        replay_buffer.count = np.sum([len(rf.a) for rf in replay_buffers])

        total_steps += replay_buffer.count

        pbar.update(replay_buffer.count)

        logging.info("Training")

        actor_loss, critic_loss = agent.update(replay_buffer, total_steps, device=dev_optim)

        replay_buffer.count = 0

        log = {'actor_loss': actor_loss,
               'critic_loss': critic_loss,
               'total_steps': total_steps,
               'time_elapsed': (datetime.datetime.now() - time_now).seconds}

        logging.info(log)
        wandb.log(log)

        if total_steps - prev_total_steps >= args.evaluate_freq:
            reward, length = evaluate_policy(args, env_evaluate, agent, state_norm, dev_optim)

            if reward >= max_reward:
                max_reward = reward
                torch.save(agent.actor.state_dict(), f'saved_models/agent-{time_now}.pth')

            log = {**log,
                   'episode_reward': reward,
                   'episode_length': length}

            logging.info(log)
            wandb.log(log)

            torch.save({
                'total_steps': total_steps,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            }, f'checkpoints/checkpoint-{time_now}.pt')

            prev_total_steps = total_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e8), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of collectors")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
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
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['Pendulum-v1', 'BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_index = 0
    main(args, env_name=env_name[env_index], number=1, seed=10)

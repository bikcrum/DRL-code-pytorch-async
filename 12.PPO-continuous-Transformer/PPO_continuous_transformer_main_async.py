import argparse
import collections
import datetime
import logging
import os
from copy import deepcopy

import gym
import numpy as np
import ray
import torch
import tqdm
from ppo_continuous_transformer import PPO_continuous

import wandb
# from envs.templates.navigation.navigation_env import CassieNavigationEnv
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer

logging.getLogger().setLevel(logging.INFO)

ray.init(num_cpus=20, num_gpus=1, local_mode=False)


# ray.init(num_cpus=20)

# @ray.remote(num_gpus=0.1)
@ray.remote
class Evaluator:
    def __init__(self, env, state_norm, args):
        self.args = args
        self.state_norm = state_norm
        self.env = env()

    def run(self, actor, device, render=False):
        actor = actor.to(device)

        def reset_rnn_hidden():
            actor.rnn_hidden = None

        def choose_action_transformer(s, device):
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float, device=device)

                assert s.dim() == 2, "s1 must be 2D, [seq_len, state_dim]"

                # Add batch dimension
                s = s.unsqueeze(0)
                # s1: [1, seq_len, state_dim]

                mean, _ = actor(s)
                # mean: [1, seq_len, action_dim]

                # Get output from last observation
                mean = mean.squeeze(0)[-1]
                # mean: [action_dim]

                return mean.detach().numpy()

        def choose_action_rnn(s, device):
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float, device=device)

                assert s.dim() == 1, "s must be 1D, [state_dim]"

                # Add batch dimension
                s = s.unsqueeze(0)
                # s: [1, state_dim]

                mean, _ = actor(s)
                # mean: [1, action_dim]

                mean = mean.squeeze(0)
                # mean: [action_dim]

                return mean.detach().numpy()

        reward = 0
        length = 0

        s = self.env.reset()
        if self.args.use_state_norm:
            s = self.state_norm(s, update=False)  # During the evaluating,update=False

        done = False
        episode_length = 0
        episode_reward = 0
        curr_buf = collections.deque(maxlen=self.args.transformer_max_len)

        # reset_rnn_hidden()

        while not done:
            if len(curr_buf) == self.args.transformer_max_len:
                # reset_rnn_hidden()
                curr_buf.popleft()
            curr_buf.append(s)
            a = choose_action_transformer(curr_buf, device)  # We use the deterministic policy during the evaluating
            # a = choose_action_rnn(s, device)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = self.env.step(a * self.args.max_action)

            if render and not done:
                self.env.render()

            if self.args.use_state_norm:
                s_ = self.state_norm(s_, update=False)

            episode_reward += r
            episode_length += 1
            s = s_
        reward += episode_reward
        length += episode_length

        return reward, length


# @ray.remote(num_gpus=0.1)
@ray.remote
class Collector:
    def __init__(self, env, state_norm, reward_scaling, reward_norm, batch_size, args, device):
        self.env = env()
        self.state_norm = state_norm
        self.reward_scaling = reward_scaling
        self.reward_norm = reward_norm
        self.batch_size = batch_size
        self.args = args
        self.device = device

    def run(self, actor, render=False):
        def reset_rnn_hidden():
            actor.rnn_hidden = None

        def choose_action_transformer(s, device):
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float, device=device)

                assert s.dim() == 2, "s1 must be 2D, [seq_len, state_dim]"

                # Add batch dimension
                s = s.unsqueeze(0)
                # s1: [1, seq_len, state_dim]

                dist = actor.get_distribution(s)
                a = dist.sample()
                # a: [1, seq_len, action_dim]

                a_logprob = dist.log_prob(a)
                # a_logprob: [1, seq_len, action_dim]

                a, a_logprob = a.squeeze(0)[-1], a_logprob.squeeze(0)[-1]
                # a: [action_dim], a_logprob: [action_dim]

                return a.detach().numpy(), a_logprob.detach().numpy()

        def choose_action_RNN(s, device):
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float, device=device)

                assert s.dim() == 1, "s must be 1D, [state_dim]"

                # Add batch dimension
                s = s.unsqueeze(0)
                # s: [1, state_dim]

                dist = actor.get_distribution(s)
                a = dist.sample()
                # a: [1, action_dim]

                a_logprob = dist.log_prob(a)
                # a_logprob: [1, action_dim]

                a, a_logprob = a.squeeze(0), a_logprob.squeeze(0)
                # a: [action_dim], a_logprob: [action_dim]

                return a.detach().numpy(), a_logprob.detach().numpy()

        # def choose_action_transformer(buffer):
        #     s1, s2 = zip(*buffer)
        #     s1 = torch.tensor(s1, dtype=torch.float, device=self.device).unsqueeze(1)  # [S, B, A]
        #     s2 = torch.tensor(s2, dtype=torch.float, device=self.device).unsqueeze(1)  # [S, B, C, W, H]
        #     with torch.no_grad():
        #         dist = actor.get_dist(s1, s2,
        #                               mask=nn.Transformer.generate_square_subsequent_mask(s1.size(0)),
        #                               src_key_padding_mask=torch.zeros((s1.size(1), s1.size(0))),
        #                               ep_lens=len(buffer))  # (B, S)
        #         a = dist.sample()  # Sample the action according to the probability distribution
        #         a = torch.clamp(a, -1.0, 1.0)  # [-max,max]
        #         a_logprob = dist.log_prob(a)  # The log probability density of the action
        #     return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

        replay_buffer = ReplayBuffer(self.args)

        total_reward = []
        total_steps = []

        episode_reward = 0
        episode_step = 0

        s = self.env.reset()

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        # while replay_buffer.count < self.batch_size:
        for _ in range(self.args.batch_size):
            # s = self.env.reset()
            # curr_buf = []
            # if render:
            #     self.env.render()
            # if self.args.use_state_norm:
            #     s = self.state_norm(s)
            # if self.args.use_reward_scaling:
            #     self.reward_scaling.reset()
            # episode_steps = 0
            # episode_reward = 0
            # done = False

            state_buffer = []
            # reset_rnn_hidden()

            if self.args.transformer_randomize_len:
                max_seq_length = np.random.randint(1, self.args.transformer_max_len + 1)
            else:
                max_seq_length = self.args.transformer_max_len

            for seq_step in range(max_seq_length):
                if self.args.use_state_norm:
                    s = self.state_norm(s)

                state_buffer.append(s)
                # while not done and replay_buffer.count < self.args.batch_size:
                # print(f'Replay buffer size: {replay_buffer.count}')
                # if len(curr_buf) >= self.args.transformer_max_len:
                #     replay_buffer.save_trajectory()
                #     curr_buf = []

                # episode_steps += 1
                # curr_buf.append(s)
                # a, a_logprob = choose_action(s)  # Action and the corresponding log probability
                a, a_logprob = choose_action_transformer(state_buffer, self.device)
                # a, a_logprob = choose_action_RNN(s, self.device)
                s_, r, done, _ = self.env.step(a * self.args.max_action)

                if render and not done:
                    self.env.render()

                episode_reward += r
                episode_step += 1
                #
                # if self.args.use_state_norm:
                #     s_ = self.state_norm(s_)
                # if self.args.use_reward_norm:
                #     r = self.reward_norm(r)
                # elif self.args.use_reward_scaling:
                #     r = self.reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_step != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                if self.args.use_reward_scaling:
                    r = self.reward_scaling(r)

                replay_buffer.store_transition(seq_step, s, a, a_logprob, r, dw)
                s = s_

                if done or episode_step == self.args.episode_limit:
                    break

            if self.args.use_state_norm:
                s = self.state_norm(s)

            replay_buffer.store_last_state(seq_step + 1, s)

            if done or episode_step == self.args.episode_limit:
                total_reward.append(episode_reward)
                total_steps.append(episode_step)

                episode_reward = 0
                episode_step = 0

                s = self.env.reset()

                if self.args.use_reward_scaling:
                    self.reward_scaling.reset()

        return replay_buffer, np.mean(total_reward), np.mean(total_steps), np.sum(total_steps)


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


def optimizer_to_device(optimizer, device):
    state_dict = optimizer.state_dict()

    if 'state' not in state_dict:
        logging.warning(f'No state in optimizer. Not converting to {device}')
        return

    states = state_dict['state']

    for k, state in states.items():
        for key, val in state.items():
            states[k][key] = val.to(device)


def main(args, env_name, seed):
    """
    Args:
        args:
        project_name: Name of the project reported to wandb
        previous_run: If provided, checkpoint from that run will be used. If not provided, run starts from scratch
        parent_run: If provided, will be logged to wandb (Useful to group run that continued one after another).
                    If not provided, the previous run will be resumed
    """
    # assert previous_run is None or parent_run is not None, "Previous run must require parent run"
    # If previous run is provided, run_id must be provided
    max_reward = float('-inf')
    time_now = datetime.datetime.now()

    run_name = str(time_now)

    # env = gym.make(env_name)
    # env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    # env = CassieNavigationEnv()
    # env_evaluate = CassieNavigationEnv()  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # args.state_dim = env.observation_space.shape[0]
    # args.action_dim = env.action_space.shape[0]
    # args.max_action = float(env.action_space.high[0])
    # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.episode_limit = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("episode_limit={}".format(args.episode_limit))
    # print("max_action={}".format(args.max_action))
    # print("min_action={}".format(env.max_action))
    # print("max_action={}".format(env.min_action))

    epochs = 0
    total_steps = 0  # Record the total steps during the training
    dev_inf, dev_optim = get_device()

    agent = PPO_continuous(args)

    logging.info(f'Using device:{dev_inf}(inference), {dev_optim}(optimization)')

    # Create new run
    run = wandb.init(
        entity='team-osu',
        project=f'toy-test-{env_name}',
        name=run_name,
        config=args.__dict__,
        # mode='disabled'
    )

    optimizer_to_device(agent.optimizer_actor, device=dev_optim)
    optimizer_to_device(agent.optimizer_critic, device=dev_optim)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    reward_norm = None
    reward_scaling = None
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    pbar = tqdm.tqdm(total=args.max_train_steps)

    n_collectors = args.n_collectors
    _args = deepcopy(args)
    _args.batch_size //= n_collectors

    if _args.transformer_randomize_len:
        if _args.batch_size * args.transformer_max_len / 2.0 < args.episode_limit:
            logging.warning(
                f'Each collector can only collect average of {_args.batch_size * args.transformer_max_len / 2.0} '
                f'timesteps but env has maximum of {args.episode_limit} steps')
    else:
        if _args.batch_size * args.transformer_max_len < args.episode_limit:
            logging.warning(
                f'Each collector can only collect maximum of {_args.batch_size * args.transformer_max_len} '
                f'timesteps but env has maximum of {args.episode_limit} steps')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    prev_total_steps = 0

    collectors = [Collector.remote(lambda: env, state_norm, reward_scaling, reward_norm,
                                   _args.batch_size,
                                   _args,
                                   dev_inf) for _ in range(n_collectors)]

    evaluators = [Evaluator.remote(lambda: env_evaluate, state_norm, args) for _ in range(args.n_evaluators)]

    while total_steps < args.max_train_steps:
        actor = agent.actor.to(dev_inf)

        logging.info("Collecting data")
        time_collecting = datetime.datetime.now()
        replay_buffers, mean_ep_rewards, mean_ep_lens, collector_total_steps = zip(*ray.get(
            [collector.run.remote(actor, render=False) for collector in collectors]))

        time_collecting = datetime.datetime.now() - time_collecting

        mean_ep_rewards = np.array(mean_ep_rewards)
        mean_ep_lens = np.array(mean_ep_lens)
        collector_total_steps = sum(collector_total_steps)

        episode_reward = mean_ep_rewards[~np.isnan(mean_ep_rewards)].mean()
        episode_length = mean_ep_lens[~np.isnan(mean_ep_lens)].mean()
        #
        # replay_buffer.s1 = np.vstack([rf.s1 for rf in replay_buffers])
        # replay_buffer.s2 = np.vstack([rf.s2 for rf in replay_buffers])
        # replay_buffer.r = np.vstack([rf.r for rf in replay_buffers])
        # replay_buffer.a = np.vstack([rf.a for rf in replay_buffers])
        # replay_buffer.done = np.vstack([rf.done for rf in replay_buffers])
        # replay_buffer.a_logprob = np.vstack([rf.a_logprob for rf in replay_buffers])
        # replay_buffer.dw = np.vstack([rf.dw for rf in replay_buffers])
        # replay_buffer.s1_ = np.vstack([rf.s1_ for rf in replay_buffers])
        # replay_buffer.s2_ = np.vstack([rf.s2_ for rf in replay_buffers])
        # replay_buffer.trajectory_indices = []
        # replay_buffer.count = 0
        #
        # for rf in replay_buffers:
        #     replay_buffer.trajectory_indices += [idx + replay_buffer.count for idx in rf.trajectory_indices]
        #     replay_buffer.count += rf.count
        #
        # replay_buffer.trajectory_indices = np.array(replay_buffer.trajectory_indices)

        total_steps += collector_total_steps

        pbar.update(collector_total_steps)

        log = {'episode_reward': episode_reward,
               'episode_length': episode_length,
               'total_steps': total_steps,
               'epochs': epochs,
               'time_collecting': time_collecting.total_seconds(),
               'time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.info(log)
        run.log(log, step=total_steps)

        logging.info("Training")
        time_training = datetime.datetime.now()
        actor_loss, critic_loss, entropy, entropy_bonus = agent.update(replay_buffers, total_steps, device=dev_optim)
        time_training = datetime.datetime.now() - time_training

        # replay_buffer.count = 0

        log = {'actor_loss': actor_loss,
               'critic_loss': critic_loss,
               'entropy': entropy,
               'entropy_bonus': entropy_bonus,
               'total_steps': total_steps,
               'epochs': epochs,
               'time_training': time_training.total_seconds(),
               'time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.info(log)
        run.log(log, step=total_steps)

        if total_steps - prev_total_steps >= args.evaluate_freq:
            # reward, length = evaluate_policy(args, env_evaluate, agent, state_norm, dev_optim)
            actor = agent.actor.to(dev_inf)

            logging.info("Evaluating")
            time_evaluating = datetime.datetime.now()
            data = ray.get([evaluator.run.remote(actor, dev_inf) for evaluator in evaluators])
            time_evaluating = datetime.datetime.now() - time_evaluating

            reward, length = list(zip(*data))

            reward, length = np.mean(reward), np.mean(length)

            if reward >= max_reward:
                max_reward = reward
                torch.save(agent.actor.state_dict(), f'saved_models/agent-{run.name}.pth')
                run.save(f'saved_models/agent-{run.name}.pth')

            log = {'episode_reward_eval': reward,
                   'episode_length_eval': length,
                   'total_steps': total_steps,
                   'epochs': epochs,
                   'time_evaluating': time_evaluating.total_seconds(),
                   'time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

            logging.info(log)
            run.log(log, step=total_steps)

            torch.save({
                'total_steps': total_steps,
                'epochs': epochs,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            }, f'checkpoints/checkpoint-{run.name}.pt')

            run.save(f'checkpoints/checkpoint-{run.name}.pt')

            prev_total_steps = total_steps

        epochs += 1

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous-transformer")
    parser.add_argument("--max_train_steps", type=int, default=int(3e8), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--n_collectors", type=int, default=4, help="Number of collectors")
    parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--transformer_max_len", type=int, default=1600,
                        help="The maximum length of observation that transformed needed to attend backward")
    parser.add_argument('--transformer_randomize_len', type=bool, default=False, help='randomize length of sequence')
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")

    args = parser.parse_args()

    env_names = ['MountainCarContinuous-v0', 'Pendulum-v1', 'BipedalWalker-v3']
    env_index = 2

    # Create new run from scratch
    main(args, env_name=env_names[env_index], seed=0)

    # Resume previous run
    # main(args,
    #      project_name='NavigationEnv',
    #      previous_run='2023-03-26 14:03:41.463767')

    # Resume from previous run but create new run (parent_run is logged in wandb to group runs continued after one
    # another under same parent name
    # main(args,
    #      project_name='NavigationEnv',
    #      previous_run='2023-03-28 01:22:35.005293',
    #      parent_run='2023-03-28 01:22:35.005293')

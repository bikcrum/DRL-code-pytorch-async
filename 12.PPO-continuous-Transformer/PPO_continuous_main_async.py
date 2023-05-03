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
import wandb

from normalization import Normalization, RewardScaling
from ppo_continuous import PPO_continuous
from replaybuffer import ReplayBuffer

logging.getLogger().setLevel(logging.DEBUG)

# ray.init(num_cpus=20, num_gpus=1, local_mode=True)


ray.init(num_cpus=20, num_gpus=1)


# @ray.remote(num_gpus=0.1)
@ray.remote
class Worker:
    def __init__(self, env, dispatcher, actor, args, device, worker_id):
        self.env = env()
        self.dispatcher = dispatcher
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        self.args = args
        self.device = device
        self.actor = deepcopy(actor).to(device)
        self.worker_id = worker_id

    def update_model(self, new_actor_params):
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def get_action(self, s, deterministic=False):
        with torch.no_grad():
            s = np.stack(s)

            s = torch.tensor(s, dtype=torch.float, device=self.device)

            assert s.dim() == 2, f"s must be 2D, [seq_len, state_dim]. Actual: {s.dim()}"

            # Add batch dimension
            s = s.unsqueeze(0)
            # s: [1, seq_len, state_dim]

            if deterministic:
                a, _ = self.actor(s)
                # Get output from last observation

                a = a.squeeze(0)[-1]
                # mean: [action_dim]
                return a.detach().numpy()
            else:
                dist = self.actor.pdf(s)
                a = dist.sample()
                # a: [1, seq_len, action_dim]

                a_logprob = dist.log_prob(a)
                # a_logprob: [1, seq_len, action_dim]

                a, a_logprob = a.squeeze(0)[-1], a_logprob.squeeze(0)[-1]
                # a: [action_dim], a_logprob: [action_dim]

                return a.detach().numpy(), a_logprob.detach().numpy()

    def collect(self, max_ep_len, render=False):
        replay_buffer = ReplayBuffer(self.args, buffer_size=max_ep_len)

        episode_reward = 0

        s = self.env.reset()

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        state_buffer = collections.deque(maxlen=self.args.transformer_max_len)

        for step in range(max_ep_len):
            if len(state_buffer) == self.args.transformer_max_len:
                state_buffer.popleft()

            state_buffer.append(s)

            a, a_logprob = self.get_action(state_buffer, deterministic=False)
            s_, r, done, _ = self.env.step(a * self.args.max_action)

            if render and not done:
                self.env.render()

            episode_reward += r

            # if done and step != self.args.time_horizon - 1:
            if done:
                dw = True
            else:
                dw = False

            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)

            replay_buffer.store_transition(s, a, a_logprob, r, dw)

            s = s_

            if done or replay_buffer.is_full():
                break

            if not ray.get(self.dispatcher.is_collecting.remote()):
                del replay_buffer
                return

        replay_buffer.store_last_state(s)

        return replay_buffer, episode_reward, step + 1, self.worker_id

    def evaluate(self, max_ep_len, render=False):
        s = self.env.reset()
        if self.args.use_state_norm:
            s = self.state_norm(s, update=False)

        episode_reward = 0
        curr_buf = collections.deque(maxlen=self.args.transformer_max_len)

        for step in range(max_ep_len):
            if len(curr_buf) == self.args.transformer_max_len:
                curr_buf.popleft()
            curr_buf.append(s)
            a = self.get_action(curr_buf, deterministic=True)
            s_, r, done, _ = self.env.step(a * self.args.max_action)

            if render and not done:
                self.env.render()

            if self.args.use_state_norm:
                s_ = self.state_norm(s_, update=False)

            episode_reward += r
            s = s_

            if done:
                break

            if not ray.get(self.dispatcher.is_evaluating.remote()):
                return

        return None, episode_reward, step + 1, self.worker_id


@ray.remote
class Dispatcher:
    def __init__(self):
        self.collecting = False
        self.evaluating = False

    def is_collecting(self):
        return self.collecting

    def is_evaluating(self):
        return self.evaluating

    def set_collecting(self, val):
        self.collecting = val

    def set_evaluating(self, val):
        self.evaluating = val


def init_logger(agent, run_name, project_name, previous_run, parent_run):
    epochs = 0
    total_steps = 0

    # Create new run from scratch if previous run is not provided
    if previous_run is None:
        # parent_run by default is equal to run name if not provided
        if parent_run is None:
            parent_run = run_name

        run = wandb.init(
            entity='team-osu',
            project=project_name,
            name=run_name,
            # mode='disabled',
            config={**args.__dict__, 'parent_run': parent_run},
            id=run_name.replace(':', '_'),
        )
    # Previous run is given, parent run not given -> resume training
    elif parent_run is None:
        run = wandb.init(
            entity='team-osu',
            project=project_name,
            resume='allow',
            id=previous_run.replace(':', '_'),
        )

        if run.resumed:
            checkpoint = torch.load(run.restore(f'checkpoints/checkpoint-{run.name}.pt'), map_location=agent.device)
            logging.info(f'Resuming from the run: {run.name} ({run.id})')
            total_steps = checkpoint['total_steps']
            epochs = checkpoint['epochs']
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

            # optimizer_to_device(agent.optimizer_actor, device=dev_optim)
            # optimizer_to_device(agent.optimizer_critic, device=dev_optim)
        else:
            logging.error(f'Run: {previous_run} did not resume')
            raise Exception(f'Run: {previous_run} did not resume')
    # Previous run is given, parent run is given, resume training but create new run under same parent
    else:
        wandb.login()

        run = wandb.Api().run(os.path.join(project_name, previous_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {previous_run}')
        run.file(name=f'checkpoints/checkpoint-{previous_run}.pt').download(replace=True)

        with open(f'checkpoints/checkpoint-{previous_run}.pt', 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            entity='team-osu',
            project=project_name,
            name=run_name,
            config={**args.__dict__, 'parent_run': parent_run},
            id=run_name.replace(':', '_'),
        )

        total_steps = checkpoint['total_steps']
        epochs = checkpoint['epochs']
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    return run, epochs, total_steps


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


def update_model(model, new_model_params):
    for p, new_p in zip(model.parameters(), new_model_params):
        p.data.copy_(new_p)


def main(args, env_name, previous_run=None, parent_run=None):
    max_reward = float('-inf')
    time_now = datetime.datetime.now()

    run_name = str(time_now)

    env = gym.make(env_name)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.time_horizon = env._max_episode_steps
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("time_horizon={}".format(args.time_horizon))

    dev_inf, dev_optim = get_device()

    agent = PPO_continuous(args, dev_optim)

    logging.info(f'Using device:{dev_inf}(inference), {dev_optim}(optimization)')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    run, epochs, total_steps = init_logger(agent, run_name, env_name, previous_run, parent_run)

    pbar = tqdm.tqdm(total=args.max_steps)

    prev_total_steps = 0

    actor_global = deepcopy(agent.actor).to(dev_inf)
    critic_global = deepcopy(agent.critic).to(dev_inf)

    dispatcher = Dispatcher.remote()

    collectors = [Worker.remote(lambda: env, dispatcher, actor_global, args, dev_inf, i) for i in
                  range(args.n_collectors)]

    evaluators = [Worker.remote(lambda: env, dispatcher, actor_global, args, dev_inf, i) for i in
                  range(args.n_evaluators)]

    replay_buffer = ReplayBuffer(args, buffer_size=args.buffer_size)

    while total_steps < args.max_steps:
        actor_param_id = ray.put(list(actor_global.parameters()))

        evaluator_ids = []
        if total_steps - prev_total_steps >= args.evaluate_freq:
            """Evaluation"""
            logging.info("Evaluating")
            time_evaluating = datetime.datetime.now()

            # Copy the latest actor to all evaluators
            for evaluator in evaluators:
                evaluator.update_model.remote(actor_param_id)

            # Evaluate policy
            ray.get(dispatcher.set_evaluating.remote(True))
            evaluator_ids = [
                evaluator.evaluate.remote(max_ep_len=min(args.time_horizon, args.buffer_size), render=False) for
                evaluator in evaluators]

            prev_total_steps = total_steps

        """Collect data"""
        logging.info("Collecting")
        time_collecting = datetime.datetime.now()

        # Copy the latest actor to all collectors
        for collector in collectors:
            collector.update_model.remote(actor_param_id)

        # Collect data
        ray.get(dispatcher.set_collecting.remote(True))
        collector_ids = [collector.collect.remote(max_ep_len=min(args.time_horizon, args.buffer_size), render=False)
                         for collector in
                         collectors]

        evaluator_steps = 0

        eval_rewards = []
        eval_lengths = []

        train_rewards = []
        train_lengths = []

        replay_buffer.reset_buffer()

        while evaluator_ids or collector_ids:
            done_ids, remain_ids = ray.wait(collector_ids + evaluator_ids, num_returns=1)

            _replay_buffer, episode_reward, episode_length, worker_id = ray.get(done_ids[0])

            if _replay_buffer is None:
                # This worker is evaluator

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)

                evaluator_steps += episode_length

                rem_buffer_size = args.buffer_size - evaluator_steps

                if rem_buffer_size > 0:
                    logging.debug(f"{rem_buffer_size} steps remaining to evaluate")
                    evaluator_ids[worker_id] = evaluators[worker_id].evaluate.remote(
                        max_ep_len=min(args.time_horizon, rem_buffer_size), render=False)
                else:
                    time_evaluating = datetime.datetime.now() - time_evaluating
                    logging.debug('Evaluation done. Cancelling stale evaluators')
                    ray.get(dispatcher.set_evaluating.remote(False))
                    map(ray.cancel, evaluator_ids)
                    evaluator_ids.clear()
            else:
                # This worker is collector

                train_rewards.append(episode_reward)
                train_lengths.append(episode_length)

                replay_buffer.merge(_replay_buffer)

                del _replay_buffer

                if not replay_buffer.is_full():
                    logging.debug(f"{args.buffer_size - replay_buffer.count} steps remaining to collect")
                    collector_ids[worker_id] = collectors[worker_id].collect.remote(
                        max_ep_len=min(args.time_horizon, args.buffer_size - replay_buffer.count),
                        render=False)
                else:
                    time_collecting = datetime.datetime.now() - time_collecting
                    logging.debug('Collector done. Cancelling stale collectors')
                    ray.get(dispatcher.set_collecting.remote(False))
                    map(ray.cancel, collector_ids)
                    collector_ids.clear()

        if evaluator_steps:
            reward, length = np.mean(eval_rewards), np.mean(eval_lengths)

            if reward >= max_reward:
                max_reward = reward
                torch.save(agent.actor.state_dict(), f'saved_models/agent-{run.name}.pth')
                run.save(f'saved_models/agent-{run.name}.pth', policy='now')

            log = {'episode_reward_eval': reward,
                   'episode_length_eval': length,
                   'total_steps': total_steps,
                   'epochs': epochs,
                   'time_evaluating': time_evaluating.total_seconds(),
                   'time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

            logging.info(log)
            run.log(log, step=total_steps)

        # Merge buffers
        batch = ReplayBuffer.create_batch(replay_buffer, args, critic_global, dev_inf)

        train_rewards = np.array(train_rewards).mean()
        train_lens = np.array(train_lengths).mean()

        total_steps += replay_buffer.count

        pbar.update(replay_buffer.count)

        log = {'episode_reward': train_rewards,
               'episode_length': train_lens,
               'total_steps': total_steps,
               'epochs': epochs,
               'new_batch_size': batch['a'].size(0),
               'time_collecting': time_collecting.total_seconds(),
               'time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.info(log)
        run.log(log, step=total_steps)

        """Training"""
        logging.info("Training")
        time_training = datetime.datetime.now()
        actor_loss, critic_loss, entropy, entropy_bonus = agent.update(batch, total_steps, device=dev_optim)

        # Copy updated models to global models
        update_model(actor_global, agent.actor.parameters())
        update_model(critic_global, agent.critic.parameters())

        time_training = datetime.datetime.now() - time_training

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

        torch.save({
            'total_steps': total_steps,
            'epochs': epochs,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
        }, f'checkpoints/checkpoint-{run.name}.pt')

        run.save(f'checkpoints/checkpoint-{run.name}.pt', policy='now')

        epochs += 1

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous-transformer")
    parser.add_argument("--max_steps", type=int, default=int(4e9), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Policy evaluation frequency")
    parser.add_argument("--n_collectors", type=int, default=4, help="Number of collectors")
    parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    parser.add_argument("--buffer_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--eval_steps", type=int, default=8192, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layers of sequential layer neural network")
    parser.add_argument("--transformer_max_len", type=int, default=16,
                        help="The maximum length of sequence in transformer encoder")
    parser.add_argument('--transformer_num_layers', type=int, default=1, help='Number of layers in transformer encoder')
    parser.add_argument('--transformer_nhead', type=int, default=1,
                        help='Number of attention heads in transformer encoder')
    parser.add_argument('--transformer_dim_feedforward', type=int, default=64,
                        help='Feedforward dimension in transformer encoder')
    parser.add_argument('--transformer_dropout', type=int, default=0.0,
                        help='Dropout positional encoder and transformer encoder')
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--num_epoch", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--eps", type=float, default=1e-5, help="eps of Adam optimizer (default: 1e-5)")

    args = parser.parse_args()

    env_names = ['Humanoid-v4', 'HalfCheetah-v2', 'MountainCarContinuous-v0', 'Pendulum-v1', 'BipedalWalker-v3']
    env_index = 4

    main(args, env_name=env_names[env_index])

import argparse
import collections
import logging
import os.path
import select
import sys
import tty

import cv2
import gym
import torch
import tqdm
from torch import nn

import wandb
from PPO_continuous_transformer_main_async import get_device
from normalization import Normalization
from ppo_continuous_transformer import Actor_Transformer

logging.basicConfig(level=logging.INFO)


def has_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def evaluate_policy(env_name, run_name, replace=True, best=True, seed=0):
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous-transformer")
    # parser.add_argument("--max_train_steps", type=int, default=int(3e8), help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=float, default=5e3,
    #                     help="Evaluate the policy every 'evaluate_freq' steps")
    # # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    # parser.add_argument("--n_collectors", type=int, default=4, help="Number of collectors")
    # parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    # parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    # parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    # parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--transformer_max_len", type=int, default=1,
                        help="The maximum length of observation that transformed needed to attend backward")
    parser.add_argument('--transformer_num_layers', type=int, default=1, help='Number of layers in transformer encoder')
    parser.add_argument('--nhead', type=int, default=1, help='Number of attention heads in transformer encoder')
    # parser.add_argument('--transformer_randomize_len', type=bool, de
    # fault=False, help='randomize length of sequence')
    # parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    # parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    # parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    # parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    # parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    # parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    # parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    # parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")

    args = parser.parse_args()

    env = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    # seed = 10

    # np.random.seed(seed)
    # torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.episode_limit = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("episode_limit={}".format(args.episode_limit))

    dev_inf, dev_optim = get_device()

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization

    actor = Actor_Transformer(args)

    wandb.login()

    run = wandb.Api().run(os.path.join(f'toy-test-{env_name}', run_name.replace(':', '_')))

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    logging.info(f'Checkpoint loaded from: {run_name}')
    if best:
        if replace or not os.path.exists(f'saved_models/agent-{run_name}.pth'):
            run.file(name=f'saved_models/agent-{run_name}.pth').download(replace=replace)

        with open(f'saved_models/agent-{run_name}.pth', 'rb') as r:
            checkpoint = torch.load(r, map_location=dev_inf)

        actor.load_state_dict(checkpoint)
    else:
        if replace or not os.path.exists(f'checkpoints/checkpoint-{run_name}.pt'):
            run.file(name=f'checkpoints/checkpoint-{run_name}.pt').download(replace=replace)

        with open(f'checkpoints/checkpoint-{run_name}.pt', 'rb') as r:
            checkpoint = torch.load(r, map_location=dev_inf)

        actor.load_state_dict(checkpoint['actor_state_dict'])
    # if best:
    #     ckpt = torch.load(f'saved_models/agent-{run_name}.pth', map_location=dev_inf)
    #     agent.actor.load_state_dict(ckpt)
    # else:
    #     ckpt = torch.load(f'checkpoints/checkpoint-{run_name}.pt', map_location=dev_inf)
    #     agent.actor.load_state_dict(ckpt['actor_state_dict'])

    render = True

    n_epoch = 50
    reward = 0
    length = 0

    tty.setcbreak(sys.stdin.fileno())

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

        # s: [batch_size, seq_len, state_dim], ep_lens: [batch_size]

    def actor_forward(s):
        assert s.dim() == 3, "Actor_Transformer only accept 3d input. [batch_size, seq_len, state_dim]"

        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, state_dim]

        s = torch.relu(actor.actor_fc1(s))
        # s: [seq_len, batch_size, hidden_dim]

        s = actor.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim]

        mask = nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(s.device)
        s, attn_maps = actor.transformer_encoder(s, mask=mask, need_weights=True)
        # s: [seq_len, batch_size, hidden_dim]

        attn_maps = torch.stack(attn_maps)
        # attn_maps: [num_layers, batch_size, seq_len, seq_len]

        # logit = self.actor_fc2(s)
        # logit: [seq_len, batch_size, action_dim]

        logit = s.transpose(0, 1)
        # logits: [batch_size, seq_len, action_dim]

        # Tanh because log_std range is [-1, 1]
        mean = torch.tanh(actor.mean_layer(logit))
        # mean: [batch_size, seq_len, action_dim]

        # Tanh because log_std range is [-1, 1]
        # log_std = torch.tanh(actor.log_std_layer(logit))
        log_std = actor.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        # log_std: [batch_size, seq_len, action_dim]

        return mean, log_std, attn_maps

    def choose_action_transformer(s, device):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float, device=device)

            assert s.dim() == 2, "s1 must be 2D, [seq_len, state_dim]"

            # Add batch dimension
            s = s.unsqueeze(0)
            # s1: [1, seq_len, state_dim]

            mean, _, attn_maps = actor_forward(s)
            # mean: [1, seq_len, action_dim], attn_maps: [num_layers, 1, seq_len, seq_len]

            # Get output from last observation
            mean = mean.squeeze(0)[-1]
            # mean: [action_dim]

            return mean.detach().numpy(), attn_maps[0].squeeze(0).detach().numpy()

    def choose_action_transformer_sample(s, device):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32, device=device)

            assert s.dim() == 2, "s1 must be 2D, [seq_len, state_dim]"

            # Add batch dimension
            s = s.unsqueeze(0)
            # s1: [1, seq_len, state_dim]

            dist = actor.get_distribution(s)
            a = dist.sample()
            # a: [1, seq_len, action_dim]

            a = a.squeeze(0)[-1]
            # a: [action_dim], a_logprob: [action_dim]

            return a.detach().numpy()

    for _ in tqdm.tqdm(range(n_epoch)):
        s = env.reset()
        # agent.actor.rnn_hidden = None

        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False

        done = False
        episode_length = 0
        episode_reward = 0

        state_buffer = collections.deque(maxlen=args.transformer_max_len)

        while not done:

            if len(state_buffer) == args.transformer_max_len:
                state_buffer.popleft()
            state_buffer.append(s)
            # a = choose_action_rnn(s, dev_inf)
            a, attn_map = choose_action_transformer(state_buffer, dev_inf)
            # a = choose_action_transformer_sample(state_buffer, dev_inf)
            # logging.info(f'Action:{a}')
            s_, r, done, info = env.step(a * args.max_action)

            if render and not done:
                env.render()

            cv2.imshow('attn_map', cv2.resize(attn_map, (256, 256), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            episode_length += 1
            s = s_

        logging.info(f'Reward:{episode_reward}')

        reward += episode_reward
        length += episode_length

    return reward / n_epoch, length / n_epoch


if __name__ == '__main__':
    # evaluate_policy(run_name='2023-03-18 18:42:08.637075')
    # evaluate_policy(run_name='2023-03-19 23:32:54.522117')
    # evaluate_policy(run_name='2023-03-20 13:17:36.096833')

    env_names = ['Humanoid-v4', 'HalfCheetah-v2', 'MountainCarContinuous-v0', 'Pendulum-v1', 'BipedalWalker-v3']
    env_index = 0

    evaluate_policy(env_name=env_names[env_index],
                    run_name='2023-04-19 02:58:04.217244',
                    replace=True,
                    best=True)
    # random()

    # cv2.imshow('rgb_image', cv2.resize(rgb_image, (320, 320)))
    # cv2.imshow('depth_map', cv2.resize(depth_map, (320, 320)))
    # cv2.imshow('follow', env.get_rgb_image(320, 320, env.vis_depth1, env.sim)[..., ::-1])
    # cv2.waitKey(1)

import argparse
import collections
import datetime
import logging
import os.path
import select
import sys
import tty
from functools import reduce

import cv2
import numpy as np
import torch
import tqdm
import wandb
from torch import nn

from envs.templates.navigation.env_training.PPO_continuous_main_async import get_device
from envs.templates.navigation.env_training_transformer.ppo_continuous import Actor_Transformer
from envs.templates.navigation.navigation_env_1 import NavigationEnv
from normalization import Normalization

logging.basicConfig(level=logging.INFO)


def random():
    env = NavigationEnv()

    render = True

    state = env.reset()

    step_time = []

    ep_reward = 0
    total = 20
    for i in tqdm.tqdm(range(total)):
        action = np.random.uniform(-1.0, 1.0, env.action_size)

        start = datetime.datetime.now()

        (state, depth), reward, done, info = env.step(action, base_render=render)

        step_time.append(datetime.datetime.now() - start)

        logging.info(f'Step time:{step_time[-1].total_seconds()}')
        #
        if render:
            env.show_depth(depth[0])

        ep_reward += reward

        # info = {'i': i, 'state': state, 'reward': reward, 'done': done, 'info': info}
        # logging.info(info)

        if done:
            logging.info(f'Episode reward={ep_reward}')
            ep_reward = 0
            if render:
                env.render()
            env.reset()

    print(f'Average step time:{reduce(lambda acc, x: acc + x, step_time).total_seconds() / total}')


def has_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def actor_forward(actor, s1, s2):
    s1 = actor.fc_layers(s1)
    # s1: [batch_size, seq_len, hidden_dim]

    B, S, C, W, H = s2.size()
    s2 = actor.conv_layers(s2.reshape(B * S, C, W, H)).reshape(B, S, -1)
    # s2: [batch_size, seq_len, hidden_dim]

    s = torch.relu(torch.cat([s1, s2], dim=-1))
    # s: [batch_size, seq_len, hidden_dim * 2]

    s = s.transpose(0, 1)
    # s: [seq_len, batch_size, hidden_dim * 2]

    s = actor.pos_encoder(s)
    # s: [seq_len, batch_size, hidden_dim * 2]

    s, attn_maps = actor.transformer_encoder(s, mask=nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(
        s.device), need_weights=True)
    # s: [seq_len, batch_size, hidden_dim * 2]

    attn_maps = torch.stack(attn_maps)
    # attn_maps: [n_layers, seq_len, batch_size, seq_len]

    s = s.transpose(0, 1)
    # s: [batch_size, seq_len, hidden_dim * 2]

    mean = torch.tanh(actor.mean_layer(s))
    # mean: [batch_size, seq_len, action_dim]

    log_std = torch.tanh(actor.std_layer(s))
    # log_std: [batch_size, seq_len, action_dim]

    return mean, log_std, attn_maps


def choose_action_transformer(actor, s, device):
    s1, s2 = zip(*s)

    with torch.no_grad():
        s1 = np.stack(s1)
        s2 = np.stack(s2)

        s1 = torch.tensor(s1, dtype=torch.float32, device=device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=device)

        assert s1.dim() == 2, f"s1 must be 2D, [seq_len, state_dim]. Actual: {s1.dim()}"
        assert s2.dim() == 4, f"s2 must be 4D, [seq_len, 1, 64, 64]. Actual: {s2.dim()}"

        # Add batch dimension
        s1 = s1.unsqueeze(0)
        # s1: [1, seq_len, state_dim]

        s2 = s2.unsqueeze(0)
        # s2: [1, seq_len, 1, 64, 64]

        mean, _, attn_maps = actor_forward(actor, s1, s2)
        # mean: [1, seq_len, action_dim]

        # Get output from last observation
        mean = mean.squeeze(0)[-1]
        # mean: [action_dim]

        return mean.detach().numpy(), attn_maps[0].squeeze(0).detach().numpy()


def evaluate_policy(project_name, run_name, replace=True, best=True):
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    # parser.add_argument("--max_steps", type=int, default=int(3e8), help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=float, default=5e3,
    #                     help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    # parser.add_argument("--n_workers", type=int, default=4, help="Number of collectors")
    # parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    # parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    # parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--transformer_max_len", type=int, default=8,
                        help="The maximum length of observation that transformed needed to attend backward")
    parser.add_argument('--transformer_num_layers', type=int, default=1, help='Number of layers in transformer encoder')
    parser.add_argument('--nhead', type=int, default=1, help='Number of attention heads in transformer encoder')
    # parser.add_argument('--transformer_randomize_len', type=bool, default=False, help='randomize length of sequence')
    # parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    # parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    # parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    # parser.add_argument("--num_epoch", type=int, default=10, help="PPO parameter")
    # parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    # parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    # parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    # parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    # parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    # parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    # max_reward = float('-inf')
    # time_now = datetime.datetime.now()

    env = NavigationEnv()  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    # seed = 10

    # np.random.seed(seed)
    # torch.manual_seed(seed)

    args.state_dim = env.observation_size
    args.action_dim = env.action_size
    args.max_episode_steps = env.max_steps  # Maximum number of steps per episode

    dev_inf, dev_optim = get_device()

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # reward_norm = None
    # reward_scaling = None
    # if args.use_reward_norm:  # Trick 3:reward normalization
    #     reward_norm = Normalization(shape=1)
    # elif args.use_reward_scaling:  # Trick 4:reward scaling
    #     reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    actor = Actor_Transformer(args)

    wandb.login()

    run = wandb.Api().run(os.path.join(project_name, run_name.replace(':', '_')))

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

    for _ in tqdm.tqdm(range(n_epoch)):

        if render:
            env.render()
        s = env.reset()
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
            # a = agent.evaluate(s, dev_inf)  # We use the deterministic policy during the evaluating
            a, attn_map = choose_action_transformer(actor, state_buffer, dev_inf)
            # if args.policy_dist == "Beta":
            #     action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            # else:
            #     action = a
            # logging.info(f'Action:{a}')
            s_, r, done, info = env.step(a, base_render=render)
            # logging.info(info)

            if render and not done:
                env.render()

                cv2.imshow('attn_map', cv2.resize(attn_map, (256, 256), interpolation=cv2.INTER_NEAREST))

                env.show_depth(s_[1])

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            episode_length += 1
            s = s_

            if has_data():
                c = sys.stdin.read(1)
                # Keystroke for reset
                if c == 'r':
                    break
                elif c in ['a', 'd', 'w', 's']:
                    if c == 'a':
                        shift = np.array([[0.0, -0.5]])
                    elif c == 'd':
                        shift = np.array([[0.0, 0.5]])
                    elif c == 'w':
                        shift = np.array([[-0.5, 0.0]])
                    else:
                        shift = np.array([[0.5, 0.0]])

                    env.obstacle_positions = env.obstacle_positions + shift
                    env.set_obstacles_position(env.sim, env.obstacle_positions)
                    env.obstacles_bbox = env.get_obstacle_bbox(env.sim)

                    # env.goal_position = env.goal_position + shift[0]
                    # env.set_goal_position(env.sim, env.goal_position)

        logging.info(f'Reward:{episode_reward}')

        reward += episode_reward
        length += episode_length

    return reward / n_epoch, length / n_epoch


if __name__ == '__main__':
    # evaluate_policy(run_name='2023-03-18 18:42:08.637075')
    # evaluate_policy(run_name='2023-03-19 23:32:54.522117')
    # evaluate_policy(run_name='2023-03-20 13:17:36.096833')
    evaluate_policy(project_name='NavigationEnv_Transformer',
                    run_name='2023-04-19 22:31:12.572387',
                    replace=True,
                    best=False)
    # random()

    # cv2.imshow('rgb_image', cv2.resize(rgb_image, (320, 320)))
    # cv2.imshow('depth_map', cv2.resize(depth_map, (320, 320)))
    # cv2.imshow('follow', env.get_rgb_image(320, 320, env.vis_depth1, env.sim)[..., ::-1])
    # cv2.waitKey(1)

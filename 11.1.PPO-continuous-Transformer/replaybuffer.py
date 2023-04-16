import logging

import torch
import numpy as np
import copy


class ReplayBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit + 1, self.state_dim]),
                       # 'v': np.zeros([self.batch_size, self.episode_limit + 1]),
                       'a': np.zeros([self.batch_size, self.episode_limit]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'dw': np.ones([self.batch_size, self.episode_limit]),
                       # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, a, a_logprob, r, dw):
        self.buffer['s'][self.episode_num][episode_step] = s
        # self.buffer['v'][self.episode_num][episode_step] = v
        self.buffer['a'][self.episode_num][episode_step] = a
        self.buffer['a_logprob'][self.episode_num][episode_step] = a_logprob
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    #
    # def store_last_value(self, episode_step, v):
    #     self.buffer['v'][self.episode_num][episode_step] = v
    #     self.episode_num += 1
    #     # Record max_episode_len
    #     if episode_step > self.max_episode_len:
    #         self.max_episode_len = episode_step

    def store_last_state(self, episode_step, s):
        self.buffer['s'][self.episode_num][episode_step] = s
        # self.buffer['v'][self.episode_num][episode_step] = v
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_adv(self, v, v_next, r, dw, active):
        # Calculate the advantage using GAE
        # v = self.buffer['v'][:, :self.max_episode_len]
        # v_next = self.buffer['v'][:, 1:self.max_episode_len + 1]
        # r = self.buffer['r'][:, :self.max_episode_len]
        # dw = self.buffer['dw'][:, :self.max_episode_len]
        # active = self.buffer['active'][:, :self.max_episode_len]
        adv = torch.zeros_like(r, device=r.device)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * ~dw - v
            for t in reversed(range(self.max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                # adv_copy = copy.deepcopy(adv)
                # adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
                # adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                adv_copy = adv.clone()
                adv_copy[active == 0] = torch.nan
                mean = torch.nanmean(adv_copy)
                std = torch.tensor(np.nanstd(adv_copy.cpu().numpy()), device=adv_copy.device) + 1e-5
                adv = (adv - mean) / std
        return adv, v_target

    def get_training_data(self, device, value_function):
        # active = torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.float32, device=device)
        # ep_lens = active.sum(-1).long()

        active = torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.bool, device=device)
        s = torch.tensor(self.buffer['s'][:, :self.max_episode_len + 1], dtype=torch.float32, device=device)
        # v = torch.tensor(self.buffer['v'][:, :self.max_episode_len], dtype=torch.float32, device=device)
        # v_next = torch.tensor(self.buffer['v'][:, 1:self.max_episode_len + 1], dtype=torch.float32, device=device)

        # torch.manual_seed(0)
        v = value_function(s[:, :-1]).squeeze(-1)

        # torch.manual_seed(0)
        v_last = value_function(s[:, 1:])[torch.arange(s.size(0)), -1]
        v_next = torch.cat((v[:, 1:], v_last), dim=-1)

        v[~active] = 0
        v_next[~active] = 0

        s = s[:, :-1]
        a = torch.tensor(self.buffer['a'][:, :self.max_episode_len], dtype=torch.long, device=device)
        a_logprob = torch.tensor(self.buffer['a_logprob'][:, :self.max_episode_len], dtype=torch.float32,
                                 device=device)
        r = torch.tensor(self.buffer['r'][:, :self.max_episode_len], dtype=torch.float32, device=device)
        dw = torch.tensor(self.buffer['dw'][:, :self.max_episode_len], dtype=torch.bool, device=device)

        # v_pred[~active] = 0
        # v_next_pred[~active] = 0

        # v = v_pred
        # v_next = v_next_pred

        # logging.info('prediction v: {}'.format(torch.mean(torch.abs(v - v_pred)[active])))
        # logging.info('prediction v_next: {}'.format(torch.mean(torch.abs(v_next - v_next_pred)[active])))

        adv, v_target = self.get_adv(v, v_next, r, dw, active)

        batch = dict(s=s, a=a, a_logprob=a_logprob, active=active, adv=adv, v_target=v_target)
        # batch = {'s': torch.tensor(self.buffer['s'][:, :self.max_episode_len], dtype=torch.float32, device=device),
        #          'a': torch.tensor(self.buffer['a'][:, :self.max_episode_len], dtype=torch.long, device=device),
        #          # 动作a的类型必须是long
        #          'a_logprob': torch.tensor(self.buffer['a_logprob'][:, :self.max_episode_len], dtype=torch.float32,
        #                                    device=device),
        #          'active': torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.float32,
        #                                 device=device),
        #          'adv': torch.tensor(adv, dtype=torch.float32, device=device),
        #          'v_target': torch.tensor(v_target, dtype=torch.float32, device=device)}

        return batch

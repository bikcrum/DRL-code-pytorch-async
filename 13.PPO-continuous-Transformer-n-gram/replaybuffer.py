import logging

import torch
import numpy as np
import copy
from torch.nn import functional as F


class ReplayBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        # self.transformer_max_len = args.transformer_max_len
        self.batch_size = args.batch_size
        self.count = 0
        # self.max_episode_len = 0
        # self.buffer = None
        self.ep_lens = []
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.state_dim], dtype=np.float32),
                       # 'v': np.zeros([self.batch_size]),
                       'a': np.zeros([self.batch_size, self.action_dim], dtype=np.float32),
                       'a_logprob': np.zeros([self.batch_size, self.action_dim], dtype=np.float32),
                       'r': np.zeros([self.batch_size], dtype=np.float32),
                       'dw': np.ones([self.batch_size], dtype=bool),
                       # Note: We use 'np.ones' to initialize 'dw'
                       # 'active': np.zeros([self.batch_size])
                       }
        self.count = 0
        self.s_last = []
        # self.v_last = []
        self.max_episode_len = 0

    def store_transition(self, s, a, a_logprob, r, dw):
        self.buffer['s'][self.count] = s
        # self.buffer['v'][self.count] = v
        self.buffer['a'][self.count] = a
        self.buffer['a_logprob'][self.count] = a_logprob
        self.buffer['r'][self.count] = r
        self.buffer['dw'][self.count] = dw

        self.count += 1

        # self.buffer['active'][self.count] = 1.0

    #
    # def store_last_value(self, episode_step, v):
    #     self.buffer['v'][self.episode_num][episode_step] = v
    #     self.episode_num += 1
    #     # Record max_episode_len
    #     if episode_step > self.max_episode_len:
    #         self.max_episode_len = episode_step

    def store_last_state(self, s):
        self.s_last.append(s)
        # self.v_last.append(v)
        # self.buffer['s'][self.count] = s
        # self.buffer['v'][self.episode_num][episode_step] = v
        # self.count += 1

        self.ep_lens.append(self.count)

        # Record max_episode_len
        # if episode_step > self.max_episode_len:
        #     self.max_episode_len = episode_step

    @staticmethod
    def get_adv(v, v_next, r, dw, args):
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
            deltas = r + args.gamma * v_next * ~dw - v
            for t in reversed(range(r.size(0))):
                gae = deltas[t] + args.gamma * args.lamda * gae  # gae.shape=(batch_size)
                adv[t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if args.use_adv_norm:  # Trick 1:advantage normalization
                # adv_copy = copy.deepcopy(adv)
                # adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
                # adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                mean = torch.nanmean(adv)
                std = torch.tensor(np.nanstd(adv.cpu().numpy()), device=adv.device) + 1e-5
                adv = (adv - mean) / std
        return adv, v_target

    @staticmethod
    def unfold(x, size, step):
        return x.unfold(dimension=0, size=size, step=step).permute(0, -1, *torch.arange(1, x.dim()))

    @staticmethod
    def pad_sequence(x, length):
        return F.pad(x, [0] * (x.dim() - 2) * 2 + [0, length], value=0)

    @staticmethod
    def get_training_data(s, s_last, a, a_logprob, r, dw, ep_lens, args, action_function, value_function, device):
        # active = torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.float32, device=device)
        # ep_lens = active.sum(-1).long()

        # active = torch.tensor(active, dtype=torch.bool, device=device)
        s = torch.tensor(s, dtype=torch.float32, device=device)
        s_batch = s.split(ep_lens)

        # v = torch.tensor(v, dtype=torch.float32, device=device)
        # v_batch = v.split(ep_lens)

        s_last = torch.tensor(s_last, dtype=torch.float32, device=device)
        # v_last = torch.tensor(v_last, dtype=torch.float32, device=device)

        a = torch.tensor(a, dtype=torch.float32, device=device)
        a_batch = a.split(ep_lens)

        a_logprob = torch.tensor(a_logprob, dtype=torch.float32, device=device)
        a_logprob_batch = a_logprob.split(ep_lens)

        r = torch.tensor(r, dtype=torch.float32, device=device)
        r_batch = r.split(ep_lens)

        dw = torch.tensor(dw, dtype=torch.bool, device=device)
        dw_batch = dw.split(ep_lens)

        s_batch_unfolded = []
        # v_batch_unfolded = []
        # v_next_batch_unfolded = []
        a_batch_unfolded = []
        a_logprob_batch_unfolded = []
        # r_batch_unfolded = []
        # dw_batch_unfolded = []
        active_batch_unfolded = []
        adv_batch_unfolded = []
        v_target_batch_unfolded = []

        stride = 1
        for i in range(len(s_batch)):
            seq_len = min(args.transformer_max_len, s_batch[i].size(0))

            _s = ReplayBuffer.unfold(torch.vstack((s_batch[i], s_last[i])), size=seq_len, step=stride)
            _v = value_function(_s).squeeze(-1)
            _v = torch.cat((_v[0], _v[1:, -1]))
            _v_next = _v[1:]
            _v = _v[:-1]
            _s = _s[:-1]
            _a = a_batch[i]
            _a_logprob = a_logprob_batch[i]
            _r = r_batch[i]
            _dw = dw_batch[i]

            _adv, _v_target = ReplayBuffer.get_adv(_v, _v_next, _r, _dw, args)

            _a = ReplayBuffer.unfold(_a, size=seq_len, step=stride)
            _a_logprob = ReplayBuffer.unfold(_a_logprob, size=seq_len, step=stride)
            # _r = ReplayBuffer.unfold(_r, size=seq_len, step=stride)
            # _dw = ReplayBuffer.unfold(_dw, size=seq_len, step=stride)
            _adv = ReplayBuffer.unfold(_adv, size=seq_len, step=stride)
            _v_target = ReplayBuffer.unfold(_v_target, size=seq_len, step=stride)
            _active = torch.ones((_s.size(0), _s.size(1)), dtype=torch.bool, device=device)

            if seq_len < args.transformer_max_len:
                _s = ReplayBuffer.pad_sequence(_s, args.transformer_max_len - seq_len)
                # _v = ReplayBuffer.pad_sequence(_v, args.transformer_max_len - seq_len)
                # _v_next = ReplayBuffer.pad_sequence(_v_next, args.transformer_max_len - seq_len)
                _a = ReplayBuffer.pad_sequence(_a, args.transformer_max_len - seq_len)
                _a_logprob = ReplayBuffer.pad_sequence(_a_logprob, args.transformer_max_len - seq_len)
                # _r = ReplayBuffer.pad_sequence(_r, args.transformer_max_len - seq_len)
                # _dw = ReplayBuffer.pad_sequence(_dw, args.transformer_max_len - seq_len)
                _adv = ReplayBuffer.pad_sequence(_adv, args.transformer_max_len - seq_len)
                _v_target = ReplayBuffer.pad_sequence(_v_target, args.transformer_max_len - seq_len)
                _active = ReplayBuffer.pad_sequence(_active, args.transformer_max_len - seq_len)

            s_batch_unfolded.append(_s)
            # v_batch_unfolded.append(_v)
            # v_next_batch_unfolded.append(_v_next)
            a_batch_unfolded.append(_a)
            a_logprob_batch_unfolded.append(_a_logprob)
            # r_batch_unfolded.append(_r)
            # dw_batch_unfolded.append(_dw)
            adv_batch_unfolded.append(_adv)
            v_target_batch_unfolded.append(_v_target)
            active_batch_unfolded.append(_active)

        s = torch.concatenate(s_batch_unfolded)
        # v = torch.vstack(v_batch_unfolded)
        # v_next = torch.vstack(v_next_batch_unfolded)
        a = torch.concatenate(a_batch_unfolded)
        a_logprob = torch.concatenate(a_logprob_batch_unfolded)
        # r = torch.vstack(r_batch_unfolded)
        # dw = torch.vstack(dw_batch_unfolded)
        adv = torch.concatenate(adv_batch_unfolded)
        v_target = torch.concatenate(v_target_batch_unfolded)
        active = torch.concatenate(active_batch_unfolded)

        # adv, v_target = ReplayBuffer.get_adv(v, v_next, r, dw, active, args, v.size(1))

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

    @staticmethod
    def create_batch(replay_buffers, args, action_function, value_function, device):
        s = []
        # v = []
        s_last = []
        # v_last = []
        a = []
        a_logprob = []
        r = []
        dw = []

        # max_episode_len = 0

        ep_lens = []
        for replay_buffer in replay_buffers:
            s.append(replay_buffer.buffer['s'])
            # v.append(replay_buffer.buffer['v'])
            s_last.append(replay_buffer.s_last)
            # v_last.append(replay_buffer.v_last)
            a.append(replay_buffer.buffer['a'])
            a_logprob.append(replay_buffer.buffer['a_logprob'])
            r.append(replay_buffer.buffer['r'])
            dw.append(replay_buffer.buffer['dw'])
            # active.append(replay_buffer.buffer['active'])

            _ep_lens = np.array(replay_buffer.ep_lens)
            _ep_lens[1:] = _ep_lens[1:] - _ep_lens[:-1]

            ep_lens += _ep_lens.tolist()

            # max_episode_len = max(max_episode_len, replay_buffer.max_episode_len)

        s = np.concatenate(s)
        # v = np.concatenate(v)
        s_last = np.concatenate(s_last)
        # v_last = np.concatenate(v_last)
        a = np.concatenate(a)
        a_logprob = np.concatenate(a_logprob)
        r = np.concatenate(r)
        dw = np.concatenate(dw)
        # active = np.vstack(active)

        batch = ReplayBuffer.get_training_data(s, s_last, a, a_logprob, r, dw, ep_lens, args,
                                               action_function, value_function, device)

        return batch

import logging

import torch
import numpy as np
import copy
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


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
    def get_adv(v, v_next, r, dw, dw_0pad, active, args):
        # Calculate the advantage using GAE
        adv = torch.zeros_like(r, device=r.device)
        gae = 0
        with torch.no_grad():
            deltas = r + args.gamma * v_next * ~dw - v
            deltas_0pad = r + args.gamma * v_next * ~dw_0pad - v

            assert (deltas == deltas_0pad).all()
            for t in reversed(range(r.size(1))):
                gae = deltas[:, t] + args.gamma * args.lamda * gae
                adv[:, t] = gae
            v_target = adv + v
            if args.use_adv_norm:
                mean = adv[active].mean()
                std = adv[active].std() + 1e-8
                adv = (adv - mean) / std
        return adv, v_target

    @staticmethod
    def unfold(x, size, step):
        return x.unfold(dimension=0, size=size, step=step).permute(0, -1, *torch.arange(1, x.dim()))

    @staticmethod
    def pad_sequence(x, length):
        return F.pad(x, [0] * (x.dim() - 2) * 2 + [0, length], value=0)

    @staticmethod
    def get_training_data(s, s_last, a, a_logprob, r, dw, ep_lens, args, value_function,
                          device):
        with torch.no_grad():
            # Split tensor into episodes
            s = torch.tensor(s, dtype=torch.float32, device=device)
            s_batch = s.split(ep_lens)

            s_last = torch.tensor(s_last, dtype=torch.float32, device=device)

            a = torch.tensor(a, dtype=torch.float32, device=device)
            a_batch = a.split(ep_lens)

            a_logprob = torch.tensor(a_logprob, dtype=torch.float32, device=device)
            a_logprob_batch = a_logprob.split(ep_lens)

            r = torch.tensor(r, dtype=torch.float32, device=device)
            r_batch = r.split(ep_lens)

            dw = torch.tensor(dw, dtype=torch.bool, device=device)
            dw_batch = dw.split(ep_lens)

            v_batch = []
            v_next_batch = []

            s_batch_unfolded = []
            a_batch_unfolded = []
            a_logprob_batch_unfolded = []
            active_batch_unfolded = []
            adv_batch_unfolded = []
            v_target_batch_unfolded = []

            stride = 1
            max_seq_len = min(max(ep_lens), args.transformer_max_len)
            for i in range(len(ep_lens)):
                ep_len = ep_lens[i]

                # Add last state to the end of the episode
                _s = torch.vstack((s_batch[i], s_last[i].unsqueeze(0)))
                # _s: [ep_len + 1, *state_shape]

                # If the episode is longer than transformer_max_len then the sequence is generated with stride
                if args.transformer_max_len < ep_len + 1:
                    seq_len = args.transformer_max_len
                    # assert ep_len - seq_len + 2 >= 2
                    _s = ReplayBuffer.unfold(_s, size=seq_len, step=stride)
                    # _s: [ep_len - seq_len + 2, seq_len, *state_shape]
                    _v = value_function(_s).squeeze(-1)
                    # _v: [ep_len - seq_len + 2, seq_len]
                    _v = torch.cat((_v[0], _v[1:, -1]))
                    # _v: [ep_len + 1]
                    _v_next = _v[1:]
                    # _v_next: [ep_len]
                    _v = _v[:-1]
                    # _v: [ep_len]
                    _s = _s[:-1]
                    # _s: [ep_len - seq_len + 1, seq_len, *state_shape]
                else:
                    # If the episode is shorter or equal to transformer_max_len then there is no stride
                    # so only one sequence is generated
                    _s = _s.unsqueeze(0)
                    # _s: [1, ep_len + 1, *state_shape]
                    _v = value_function(_s).squeeze(-1)
                    # _v: [1, ep_len + 1]
                    _v_next = _v[:, 1:].squeeze(0)
                    # _v_next: [ep_len]
                    _v = _v[:, :-1].squeeze(0)
                    # _v: [ep_len]
                    _s = _s[:, :-1]
                    # _s: [1, ep_len, *state_shape]

                # Pad sequences to the same length
                if _s.size(1) < max_seq_len:
                    _s = ReplayBuffer.pad_sequence(_s, max_seq_len - _s.size(1))
                    # _s: [batch, max_seq_len, *state_shape]

                s_batch_unfolded.append(_s)

                v_batch.append(_v)
                v_next_batch.append(_v_next)

            # Pad to maximum episode length (This is not same as max_seq_len)
            v = pad_sequence(v_batch, padding_value=0, batch_first=True)
            v_next = pad_sequence(v_next_batch, padding_value=0, batch_first=True)
            r = pad_sequence(r_batch, padding_value=0, batch_first=True)
            active = torch.ones_like(dw).split(ep_lens)
            dw = pad_sequence(dw_batch, padding_value=1, batch_first=True)
            dw_0pad = pad_sequence(dw_batch, padding_value=0, batch_first=True)
            active = pad_sequence(active, padding_value=0, batch_first=True)

            # Compute advantages
            adv, v_target = ReplayBuffer.get_adv(v, v_next, r, dw, dw_0pad, active, args)

            # Get non-padded sequences
            adv = adv[active].split(ep_lens)
            v_target = v_target[active].split(ep_lens)
            active = active[active].split(ep_lens)

            for i in range(len(ep_lens)):
                seq_len = min(args.transformer_max_len, ep_lens[i])

                _a = ReplayBuffer.unfold(a_batch[i], size=seq_len, step=1)
                _a_logprob = ReplayBuffer.unfold(a_logprob_batch[i], size=seq_len, step=1)
                _adv = ReplayBuffer.unfold(adv[i], size=seq_len, step=1)
                _v_target = ReplayBuffer.unfold(v_target[i], size=seq_len, step=1)
                _active = ReplayBuffer.unfold(active[i], size=seq_len, step=1)

                # Pad to max_seq_len
                if seq_len < max_seq_len:
                    _a = ReplayBuffer.pad_sequence(_a, max_seq_len - seq_len)
                    _a_logprob = ReplayBuffer.pad_sequence(_a_logprob, max_seq_len - seq_len)
                    _adv = ReplayBuffer.pad_sequence(_adv, max_seq_len - seq_len)
                    _v_target = ReplayBuffer.pad_sequence(_v_target, max_seq_len - seq_len)
                    _active = ReplayBuffer.pad_sequence(_active, max_seq_len - seq_len)

                a_batch_unfolded.append(_a)
                a_logprob_batch_unfolded.append(_a_logprob)
                adv_batch_unfolded.append(_adv)
                v_target_batch_unfolded.append(_v_target)
                active_batch_unfolded.append(_active)

            # Merge batches of sequences
            s = torch.concatenate(s_batch_unfolded)
            a = torch.concatenate(a_batch_unfolded)
            a_logprob = torch.concatenate(a_logprob_batch_unfolded)
            adv = torch.concatenate(adv_batch_unfolded)
            v_target = torch.concatenate(v_target_batch_unfolded)
            active = torch.concatenate(active_batch_unfolded)

            del s_batch_unfolded, a_batch_unfolded, a_logprob_batch_unfolded, adv_batch_unfolded, v_target_batch_unfolded, active_batch_unfolded

            batch = dict(s=s, a=a, a_logprob=a_logprob, adv=adv, v_target=v_target, active=active)

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
                                               value_function, device)

        return batch

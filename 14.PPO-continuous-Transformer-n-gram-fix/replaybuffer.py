import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, args, buffer_size):
        self.args = args
        self.buffer_size = buffer_size
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.buffer_size, self.args.state_dim], dtype=np.float32),
                       'a': np.zeros([self.buffer_size, self.args.action_dim], dtype=np.float32),
                       'a_logprob': np.zeros([self.buffer_size, self.args.action_dim], dtype=np.float32),
                       'r': np.zeros([self.buffer_size], dtype=np.float32),
                       'dw': np.ones([self.buffer_size], dtype=bool)}
        self.count = 0
        self.s_last = []
        self.ep_lens = []

    def store_transition(self, s, a, a_logprob, r, dw):
        self.buffer['s'][self.count] = s
        self.buffer['a'][self.count] = a
        self.buffer['a_logprob'][self.count] = a_logprob
        self.buffer['r'][self.count] = r
        self.buffer['dw'][self.count] = dw

        self.count += 1

    def store_last_state(self, s):
        self.s_last.append(s)

        self.ep_lens.append(self.count)

    def merge(self, replay_buffer):
        rem_count = self.buffer_size - self.count

        rem_count = min(rem_count, replay_buffer.count)

        self.buffer['s'][self.count:self.count + rem_count] = replay_buffer.buffer['s'][:rem_count]
        self.buffer['a'][self.count:self.count + rem_count] = replay_buffer.buffer['a'][:rem_count]
        self.buffer['r'][self.count:self.count + rem_count] = replay_buffer.buffer['r'][:rem_count]
        self.buffer['dw'][self.count:self.count + rem_count] = replay_buffer.buffer['dw'][:rem_count]

        self.s_last.append(replay_buffer.s_last)

        self.count += rem_count

        self.ep_lens.append(rem_count)

    def is_full(self):
        return self.count >= self.args.buffer_size

    @staticmethod
    def get_adv(v, v_next, r, dw, active, args):
        # Calculate the advantage using GAE
        adv = torch.zeros_like(r, device=r.device)
        gae = 0
        with torch.no_grad():
            deltas = r + args.gamma * v_next * ~dw - v
            for t in reversed(range(r.size(0))):
                gae = deltas[t] + args.gamma * args.lamda * gae
                adv[t] = gae
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
            v = pad_sequence(v_batch, padding_value=0)
            v_next = pad_sequence(v_next_batch, padding_value=0)
            r = pad_sequence(r_batch, padding_value=0)
            active = torch.ones_like(dw).split(ep_lens)
            dw = pad_sequence(dw_batch, padding_value=0)
            active = pad_sequence(active, padding_value=0)

            # Compute advantages
            adv, v_target = ReplayBuffer.get_adv(v, v_next, r, dw, active, args)

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
    def create_batch(replay_buffer, args, value_function, device):
        with torch.no_grad():
            s = replay_buffer.buffer['s']
            s_last = np.vstack(replay_buffer.s_last)
            a = replay_buffer.buffer['a']
            a_logprob = replay_buffer.buffer['a_logprob']
            r = replay_buffer.buffer['r']
            dw = replay_buffer.buffer['dw']
            ep_lens = replay_buffer.ep_lens

            batch = ReplayBuffer.get_training_data(s, s_last, a, a_logprob, r, dw, ep_lens, args,
                                                   value_function, device)

            return batch

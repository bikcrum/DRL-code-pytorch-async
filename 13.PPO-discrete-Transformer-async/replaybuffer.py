import torch
import numpy as np
import copy
import torch.nn.functional as F


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
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v': np.zeros([self.batch_size, self.episode_limit + 1]),
                       'a': np.zeros([self.batch_size, self.episode_limit]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'dw': np.ones([self.batch_size, self.episode_limit]),
                       # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, v, a, a_logprob, r, dw):
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v'][self.episode_num][episode_step] = v
        self.buffer['a'][self.episode_num][episode_step] = a
        self.buffer['a_logprob'][self.episode_num][episode_step] = a_logprob
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_value(self, episode_step, v):
        self.buffer['v'][self.episode_num][episode_step] = v
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_adv(self):
        # Calculate the advantage using GAE
        v = self.buffer['v'][:, :self.max_episode_len]
        v_next = self.buffer['v'][:, 1:self.max_episode_len + 1]
        r = self.buffer['r'][:, :self.max_episode_len]
        dw = self.buffer['dw'][:, :self.max_episode_len]
        active = self.buffer['active'][:, :self.max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * (1 - dw) - v
            for t in reversed(range(self.max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv_copy = copy.deepcopy(adv)
                adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target

    def get_training_data(self, device):
        adv, v_target = self.get_adv()
        batch = {'s': torch.tensor(self.buffer['s'][:, :self.max_episode_len], dtype=torch.float32, device=device),
                 'a': torch.tensor(self.buffer['a'][:, :self.max_episode_len], dtype=torch.long, device=device),
                 # 动作a的类型必须是long
                 'a_logprob': torch.tensor(self.buffer['a_logprob'][:, :self.max_episode_len], dtype=torch.float32,
                                           device=device),
                 'active': torch.tensor(self.buffer['active'][:, :self.max_episode_len], dtype=torch.float32,
                                        device=device),
                 'adv': torch.tensor(adv, dtype=torch.float32, device=device),
                 'v_target': torch.tensor(v_target, dtype=torch.float32, device=device)}

        return batch

    def get_adv_fixed_length(self, v, v_next, r, dw, active):
        # # Calculate the advantage using GAE

        adv = torch.zeros_like(r, device=r.device)
        # adv: [new_batch_size, target_len]

        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            deltas = r + self.gamma * v_next * (1 - dw) - v
            # deltas: [new_batch_size, target_len]

            for t in reversed(range(deltas.size(1))):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                # gae: [new_batch_size]

                adv[:, t] = gae

            v_target = adv + v
            # v_target: [new_batch_size, target_len]

            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv_copy = adv.clone()
                adv_copy[active == 0] = torch.nan
                mean = torch.nanmean(adv_copy)
                std = torch.tensor(np.nanstd(adv_copy.cpu().numpy()), device=adv_copy.device) + 1e-5
                adv = (adv - mean) / std
        return adv, v_target

    # Pad the tensor to the target length in the given dimension at the end by 0
    @staticmethod
    def pad_tensor(x, dim, target_len):
        n_dim = x.dim()

        pad = [0] * (n_dim - dim - 1) * 2
        padding_length = target_len - x.size(dim)

        x = F.pad(x, (*pad, 0, padding_length), 'constant', 0)

        return x

    def get_training_data_fixed_length(self, device, target_len):

        s_batch = []
        v_batch = []
        v_next_batch = []
        a_batch = []
        a_logprob_batch = []
        r_batch = []
        dw_batch = []
        active_batch = []

        for i in range(self.episode_num):
            episodic_len = self.buffer['active'][i].sum().astype(int)

            s = torch.tensor(self.buffer['s'][i, :episodic_len], dtype=torch.float32, device=device)
            # s: [episodic_len, state_dim]

            v = torch.tensor(self.buffer['v'][i, :episodic_len], dtype=torch.float32, device=device)
            # v: [episodic_len]

            v_next = torch.tensor(self.buffer['v'][i, 1:episodic_len + 1], dtype=torch.float32, device=device)
            # v_next: [episodic_len]

            a = torch.tensor(self.buffer['a'][i, :episodic_len], dtype=torch.long, device=device)
            # a: [episodic_len]

            a_logprob = torch.tensor(self.buffer['a_logprob'][i, :episodic_len], dtype=torch.float32, device=device)
            # a_logprob: [episodic_len]

            r = torch.tensor(self.buffer['r'][i, :episodic_len], dtype=torch.float32, device=device)
            # r: [episodic_len]

            dw = torch.tensor(self.buffer['dw'][i, :episodic_len], dtype=torch.float32, device=device)
            # dw: [episodic_len]

            active = torch.tensor(self.buffer['active'][i, :episodic_len], dtype=torch.float32, device=device)
            # active: [episodic_len]

            _target_len = min(target_len, episodic_len)

            # batch_size = episodic_len - _target_len + 1

            s = s.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, s.dim()))
            # s: [batch_size, _target_len, state_dim]
            s = self.pad_tensor(s, dim=1, target_len=target_len)
            # s: [batch_size, target_len, state_dim]

            v = v.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, v.dim()))
            # v: [batch_size, _target_len]
            v = self.pad_tensor(v, dim=1, target_len=target_len)
            # v: [batch_size, target_len]

            v_next = v_next.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, v_next.dim()))
            # v_next: [batch_size, _target_len]
            v_next = self.pad_tensor(v_next, dim=1, target_len=target_len)
            # v_next: [batch_size, target_len]

            a = a.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, a.dim()))
            # a: [batch_size, _target_len]
            a = self.pad_tensor(a, dim=1, target_len=target_len)
            # a: [batch_size, target_len]

            a_logprob = a_logprob.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1,
                                                                                                             a_logprob.dim()))
            # a_logprob: [batch_size, _target_len]
            a_logprob = self.pad_tensor(a_logprob, dim=1, target_len=target_len)
            # a_logprob: [batch_size, target_len]

            r = r.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, r.dim()))
            # r: [batch_size, _target_len]
            r = self.pad_tensor(r, dim=1, target_len=target_len)
            # r: [batch_size, target_len]

            dw = dw.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, dw.dim()))
            # dw: [batch_size, _target_len]
            dw = self.pad_tensor(dw, dim=1, target_len=target_len)
            # dw: [batch_size, target_len]

            active = active.unfold(dimension=0, size=_target_len, step=1).permute(0, -1, *torch.arange(1, active.dim()))
            # active: [batch_size, _target_len]
            active = self.pad_tensor(active, dim=1, target_len=target_len)
            # active: [batch_size, target_len]

            s_batch.append(s)
            v_batch.append(v)
            v_next_batch.append(v_next)
            a_batch.append(a)
            a_logprob_batch.append(a_logprob)
            r_batch.append(r)
            dw_batch.append(dw)
            active_batch.append(active)

        s_batch = torch.vstack(s_batch)
        # s_batch: [new_batch_size, target_len, state_dim]

        v_batch = torch.vstack(v_batch)
        # v_batch: [new_batch_size, target_len]

        v_next_batch = torch.vstack(v_next_batch)
        # v_next_batch: [new_batch_size, target_len]

        a_batch = torch.vstack(a_batch)
        # a_batch: [new_batch_size, target_len]

        a_logprob_batch = torch.vstack(a_logprob_batch)
        # a_logprob_batch: [new_batch_size, target_len]

        r_batch = torch.vstack(r_batch)
        # r_batch: [new_batch_size, target_len]

        dw_batch = torch.vstack(dw_batch)
        # dw_batch: [new_batch_size, target_len]

        active_batch = torch.vstack(active_batch)
        # active_batch: [new_batch_size, target_len]

        adv_batch, v_target_batch = self.get_adv_fixed_length(v=v_batch,
                                                              v_next=v_next_batch,
                                                              r=r_batch,
                                                              dw=dw_batch,
                                                              active=active_batch)
        # adv_batch: [new_batch_size, target_len]
        # v_target_batch: [new_batch_size, target_len]
        batch = {'s': s_batch,
                 'a': a_batch,
                 'a_logprob': a_logprob_batch,
                 'active': active_batch,
                 'adv': adv_batch,
                 'v_target': v_target_batch}

        return batch

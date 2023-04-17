import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from replaybuffer import ReplayBuffer


# Trick 8: orthogonal initialization


# def orthogonal_init(layer, gain=1.0):
#     nn.init.orthogonal_(layer.weight, gain=gain)
#     nn.init.constant_(layer.bias, 0)

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Actor_Transformer(nn.Module):
    def __init__(self, args):
        super(Actor_Transformer, self).__init__()

        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=0.1, max_len=args.transformer_max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)
        self.mean_layer = nn.Linear(args.hidden_dim, args.action_dim)
        self.log_std_layer = nn.Linear(args.hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.actor_fc1)
            # orthogonal_init(self.pos_encoder)
            orthogonal_init(self.mean_layer, gain=0.01)
            orthogonal_init(self.log_std_layer, gain=0.01)
            # orthogonal_init(self.critic_fc1)
            # orthogonal_init(self.critic_rnn)
            # orthogonal_init(self.critic_fc2)

    # s: [batch_size, seq_len, state_dim], ep_lens: [batch_size]
    def forward(self, s):
        assert s.dim() == 3, "Actor_Transformer only accept 3d input. [batch_size, seq_len, state_dim]"

        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, state_dim]

        s = self.actor_fc1(s)
        # s: [seq_len, batch_size, hidden_dim]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim]

        s = self.transformer_encoder(s, mask=nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(s.device))
        # s: [seq_len, batch_size, hidden_dim]

        # logit = self.actor_fc2(s)
        # logit: [seq_len, batch_size, action_dim]

        logit = s.transpose(0, 1)
        # logits: [batch_size, seq_len, action_dim]

        # Tanh because log_std range is [-1, 1]
        mean = F.tanh(self.mean_layer(logit))
        # mean: [batch_size, seq_len, action_dim]

        # Tanh because log_std range is [-1, 1]
        log_std = F.tanh(self.log_std_layer(logit))
        # log_std: [batch_size, seq_len, action_dim]

        return mean, log_std

    def get_distribution(self, s):
        mean, log_std = self.forward(s)
        # Exp to make std positive
        std = log_std.exp()
        return Normal(mean, std)


class Critic_Transformer(nn.Module):
    def __init__(self, args):
        super(Critic_Transformer, self).__init__()

        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=0.1, max_len=args.transformer_max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=4, dim_feedforward=64, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.critic_fc2 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            # orthogonal_init(self.actor_fc1)
            # orthogonal_init(self.pos_encoder)
            # orthogonal_init(self.actor_fc2, gain=0.01)
            orthogonal_init(self.critic_fc1)
            # orthogonal_init(self.critic_rnn)
            orthogonal_init(self.critic_fc2)

    # s: [batch_size, seq_len, state_dim], ep_lens: [batch_size]
    def forward(self, s):
        assert s.dim() == 3, "Critic_Transformer only accept 3d input. [batch_size, seq_len, state_dim]"

        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, state_dim]

        s = self.critic_fc1(s)
        # s: [seq_len, batch_size, hidden_dim]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim]

        s = self.transformer_encoder(s, mask=nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(s.device))
        # s: [seq_len, batch_size, hidden_dim]

        logit = self.critic_fc2(s)
        # logit: [seq_len, batch_size, 1]

        logit = logit.transpose(0, 1)
        # logits: [batch_size, seq_len, 1]

        return logit


class Actor_RNN(nn.Module):
    def __init__(self, args):
        super(Actor_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.rnn_hidden = None
        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            print("------use GRU------")
            self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            print("------use LSTM------")
            self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)

        self.mean_layer = nn.Linear(args.hidden_dim, args.action_dim)
        self.log_std_layer = nn.Linear(args.hidden_dim, args.action_dim)
        # self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.actor_fc1)
            orthogonal_init(self.actor_rnn)
            orthogonal_init(self.mean_layer, gain=0.01)
            orthogonal_init(self.log_std_layer, gain=0.01)

    # def forward(self, s):
    #     s = self.activate_func(self.actor_fc1(s))
    #     output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
    #     logit = self.actor_fc2(output)
    #     return logit

    # s: [batch_size, seq_len, state_dim], ep_lens: [batch_size]

    def to(self, device):
        if self.rnn_hidden is not None:
            self.rnn_hidden = self.rnn_hidden.to(device)
        super().to(device)
        return self

    def forward(self, s):

        s = self.activate_func(self.actor_fc1(s))
        output, self.rnn_hidden = self.actor_rnn(s, self.rnn_hidden)
        # logit = self.actor_fc2(output)

        # Tanh because log_std range is [-1, 1]
        mean = F.tanh(self.mean_layer(output))
        # mean: [batch_size, seq_len, action_dim]

        # Tanh because log_std range is [-1, 1]
        log_std = F.tanh(self.log_std_layer(output))
        # log_std: [batch_size, seq_len, action_dim]

        return mean, log_std

    def get_distribution(self, s):
        mean, log_std = self.forward(s)
        # Exp to make std positive
        std = log_std.exp()
        return Normal(mean, std)


class Critic_RNN(nn.Module):
    def __init__(self, args):
        super(Critic_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.rnn_hidden = None
        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            print("------use GRU------")
            self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            print("------use LSTM------")
            self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.critic_fc1)
            orthogonal_init(self.critic_rnn)
            orthogonal_init(self.critic_fc2)

    def to(self, device):
        if self.rnn_hidden is not None:
            self.rnn_hidden = self.rnn_hidden.to(device)
        super().to(device)
        return self

    def forward(self, s):
        s = self.activate_func(self.critic_fc1(s))
        output, self.rnn_hidden = self.critic_rnn(s, self.rnn_hidden)
        value = self.critic_fc2(output)
        return value


class PPO_continuous():
    def __init__(self, args):
        self.args = args
        # self.policy_dist = args.policy_dist
        # self.min_action = torch.tensor(args.max_action)
        # self.max_action = torch.tensor(args.max_action)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor_RNN(args)
        self.critic = Critic_RNN(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def reset_rnn_hidden(self):
        self.critic.rnn_hidden = None
        self.actor.rnn_hidden = None
    # def evaluate(self, s, device):  # When evaluating the policy, we only use the mean
    #     s1, s2 = s
    #     s1 = torch.unsqueeze(torch.tensor(s1, dtype=torch.float, device=device), 0)
    #     s2 = torch.unsqueeze(torch.tensor(s2, dtype=torch.float, device=device), 0)
    #     if self.policy_dist == "Beta":
    #         a = self.actor.mean(s).detach().cpu().numpy().flatten()
    #     else:
    #         a = self.actor(s1, s2).detach().cpu().numpy().flatten()
    #     return a

    # def choose_action(self, s):
    #     s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    #     if self.policy_dist == "Beta":
    #         with torch.no_grad():
    #             dist = self.actor.get_dist(s)
    #             a = dist.sample()  # Sample the action according to the probability distribution
    #             a_logprob = dist.log_prob(a)  # The log probability density of the action
    #     else:
    #         with torch.no_grad():
    #             dist = self.actor.get_dist(s)
    #             a = dist.sample()  # Sample the action according to the probability distribution
    #             a = torch.clamp(a, -1.0, 1.0)  # [-max,max]
    #             a_logprob = dist.log_prob(a)  # The log probability density of the action
    #     return a.numpy().flatten(), a_logprob.numpy().flatten()

    def choose_action_transformer(self, s, evaluate=False):

        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)

            assert s.dim() == 2, "s must be 2D, [seq_len, state_dim]"

            # Add batch dimension
            s = s.unsqueeze(0)
            # s: [1, seq_len, state_dim]

            if evaluate:
                mean, _ = self.actor(s)
                # mean: [1, seq_len, action_dim]

                # Get output from last observation
                mean = mean.squeeze(0)[-1]
                # mean: [action_dim]

                return mean, None
            else:
                dist = self.actor.get_distribution(s)
                a = dist.sample()
                # a: [1, seq_len, action_dim]

                a_logprob = dist.log_prob(a)
                # a_logprob: [1, seq_len, action_dim]

                a, a_logprob = a.squeeze(0)[-1], a_logprob.squeeze(0)[-1]
                # a: [action_dim], a_logprob: [action_dim]

                return a, a_logprob

    def get_value_transformer(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            value = self.critic(s)[:, -1]
            return value.item()

    def update(self, replay_buffers, total_steps, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        # self.actor.min_action = self.actor.min_action.to(device)
        # self.actor.max_action = self.actor.max_action.to(device)
        # self.min_action = self.min_action.to(device)
        # self.max_action = self.max_action.to(device)

        # s, a, a_logprob, r, s_, dw, done, ep_end_lens = replay_buffer.numpy_to_tensor(
        #     device=device)  # Get training data
        # """
        #     Calculate the advantage using GAE
        #     'dw=True' means dead or win, there is no next state s'
        #     'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        # """
        # adv = []
        # gae = 0
        # with torch.no_grad():  # adv and v_target have no gradient
        #     # Terminology:
        #     # S = episode length
        #     # B = batch size
        #
        #     s1, s2 = s
        #     # Do non-recurrent forward to get values
        #     vs = self.critic.forward_ff(s1, s2)
        #
        #     s1_, s2_ = s_
        #     # Do non-recurrent forward to get values
        #     vs_ = self.critic.forward_ff(s1_, s2_)
        #
        #     # Get sizes for each episode
        #     ep_sizes = torch.concat((ep_end_lens[:1], ep_end_lens[1:] - ep_end_lens[:-1]))
        #
        #     # Split values by episode size into a batch. We have converted collection of episode in 1-D array to
        #     # a batch of episodes
        #     vs = vs.split(ep_sizes.tolist())
        #     vs_ = vs_.split(ep_sizes.tolist())
        #
        #     # Pad all episode in a batch by value 0. (S, B, D)
        #     vs = pad_sequence(vs)
        #     vs_ = pad_sequence(vs_)
        #
        #     S, B, D = vs.size()
        #
        #     # Get true mask for padded values (B, S)
        #     src_key_padding_mask = torch.arange(S, device=device) >= ep_sizes.unsqueeze(1)
        #
        #     # Now we have a batch of episodes and each episode has observations. We do attention for each observation
        #     # with observations appearing before it (taken care by 'mask') and also indicate where padding for
        #     # episode is applied (taken care by 'src_key_padding_mask').
        #     mask = nn.Transformer.generate_square_subsequent_mask(S).to(device)
        #     vs = self.critic.forward_transformer(vs,
        #                                          mask=mask,
        #                                          src_key_padding_mask=src_key_padding_mask)
        #
        #     vs_ = self.critic.forward_transformer(vs_,
        #                                           mask=mask,
        #                                           src_key_padding_mask=src_key_padding_mask)
        #
        #     # (S, B, 1) -> (B, S, 1) -> masking -> (BxS, 1)
        #     vs = vs.permute(1, 0, 2)[~src_key_padding_mask]
        #     vs_ = vs_.permute(1, 0, 2)[~src_key_padding_mask]
        #
        #     deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
        #     for delta, d in zip(reversed(deltas.flatten()), reversed(done.flatten())):
        #         gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
        #         adv.insert(0, gae)
        #     adv = torch.tensor(adv, dtype=torch.float, device=device).view(-1, 1)
        #     v_target = adv + vs
        #     if self.use_adv_norm:  # Trick 1:advantage normalization
        #         adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        losses = []
        entropies = []
        # ep_start_indices = torch.concat((torch.LongTensor([0]).to(device), ep_end_lens))
        # Optimize policy for K epochs:

        batch = ReplayBuffer.create_batch(replay_buffers, self.args, self.critic, device)

        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                self.reset_rnn_hidden()
                dist_now = self.actor.get_distribution(
                    batch['s'][index])  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.critic(batch['s'][index]).squeeze(
                    -1)  # values_now.shape=(mini_batch_size, max_episode_len)

                dist_entropy = dist_now.entropy().sum(-1)  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index]).sum(-1)  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(
                    a_logprob_now - batch['a_logprob'][index].sum(-1))  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1,
                                        surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # critic_loss
                critic_loss = (values_now - batch['v_target'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                critic_loss = critic_loss * 0.5

                losses.append((actor_loss.item(), critic_loss.item()))

                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                mean_entropy = dist_entropy.mean().item()
                entropies.append((mean_entropy, - self.entropy_coef * mean_entropy))

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        a_loss, c_loss = zip(*losses)
        entropy, entropy_bonus = zip(*entropies)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.mean(a_loss), np.mean(c_loss), np.mean(entropy), np.mean(entropy_bonus)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


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

    # x: [seq_len, batch_size, embedding_dim]
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Actor_Transformer(nn.Module):
    def __init__(self, args):
        super(Actor_Transformer, self).__init__()

        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=args.transformer_dropout,
                                              max_len=args.transformer_max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=args.transformer_nhead,
                                                    dim_feedforward=args.transformer_dim_feedforward,
                                                    dropout=args.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.transformer_num_layers)

        self.mean_layer = nn.Linear(args.hidden_dim, args.action_dim)
        self.std_layer = nn.Linear(args.hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.actor_fc1, gain=0.01)
            orthogonal_init(self.mean_layer, gain=0.01)
            orthogonal_init(self.std_layer, gain=0.01)

    # s: [batch_size, seq_len, state_dim]
    def forward(self, s):
        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = torch.relu(self.actor_fc1(s))
        # s: [batch_size, seq_len, hidden_dim]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s, _ = self.transformer_encoder(s, mask=nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(s.device))
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = s.transpose(0, 1)
        # s: [batch_size, seq_len, hidden_dim * 2]

        mean = torch.tanh(self.mean_layer(s))
        # mean: [batch_size, seq_len, action_dim]

        std = F.softplus(self.std_layer(s))
        # std: [batch_size, seq_len, action_dim]

        return mean, std

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_Transformer(nn.Module):
    def __init__(self, args):
        super(Critic_Transformer, self).__init__()

        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=args.transformer_dropout,
                                              max_len=args.transformer_max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=args.transformer_nhead,
                                                    dim_feedforward=args.transformer_dim_feedforward,
                                                    dropout=args.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.transformer_num_layers)

        self.value_layer = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.critic_fc1, gain=0.01)
            orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s):
        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = torch.relu(self.critic_fc1(s))
        # s: [batch_size, seq_len, hidden_dim]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s, _ = self.transformer_encoder(s, mask=nn.Transformer.generate_square_subsequent_mask(s.size(0)).to(s.device))
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = self.value_layer(s)
        # mean: [batch_size, seq_len, 1]

        s = s.transpose(0, 1)
        # s: [batch_size, seq_len, 1]

        return s


class PPO_continuous:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.actor = Actor_Transformer(args)
        self.critic = Critic_Transformer(args)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a, eps=self.args.eps)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=self.args.eps)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

    def update(self, batch, total_steps, device):
        losses = []
        entropies = []

        for _ in range(self.args.num_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(batch['a'].size(0))), self.args.mini_batch_size, False):
                s = batch['s'][index].to(device)
                a = batch['a'][index].to(device)
                a_logprob = batch['a_logprob'][index].to(device)
                adv = batch['adv'][index].to(device)
                active = batch['active'][index].to(device)
                v_target = batch['v_target'][index].to(device)

                dist_now = self.actor.pdf(s)
                values_now = self.critic(s).squeeze(-1)

                dist_entropy = dist_now.entropy().sum(-1)
                a_logprob_now = dist_now.log_prob(a).sum(-1)
                ratios = torch.exp(
                    a_logprob_now - a_logprob.sum(-1))

                # actor loss
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                actor_loss = -torch.min(surr1,
                                        surr2) - self.args.entropy_coef * dist_entropy
                actor_loss = (actor_loss * active).sum() / active.sum()

                # critic_loss
                critic_loss = (values_now - v_target) ** 2
                critic_loss = (critic_loss * active).sum() / active.sum()
                critic_loss = critic_loss * 0.5

                losses.append((actor_loss.item(), critic_loss.item()))

                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                if self.args.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                mean_entropy = dist_entropy.mean().item()
                entropies.append((mean_entropy, - self.args.entropy_coef * mean_entropy))

                del s, a, a_logprob, adv, active, v_target, dist_now, values_now, dist_entropy, a_logprob_now, ratios, surr1, surr2, actor_loss, critic_loss

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

        a_loss, c_loss = zip(*losses)
        entropy, entropy_bonus = zip(*entropies)

        del losses, entropies, batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.mean(a_loss), np.mean(c_loss), np.mean(entropy), np.mean(entropy_bonus)

    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

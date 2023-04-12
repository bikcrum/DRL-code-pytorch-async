import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler

logging.getLogger().setLevel(logging.INFO)


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

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=0.1, max_len=args.episode_limit)
        encoder_layers = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            print("------use orthogonal init------")
            orthogonal_init(self.actor_fc1)
            # orthogonal_init(self.pos_encoder)
            orthogonal_init(self.actor_fc2, gain=0.01)
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

        logit = self.actor_fc2(s)
        # logit: [seq_len, batch_size, action_dim]

        logit = logit.transpose(0, 1)
        # logits: [batch_size, seq_len, action_dim]

        return logit


class Critic_Transformer(nn.Module):
    def __init__(self, args):
        super(Critic_Transformer, self).__init__()

        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=0.1, max_len=args.episode_limit)
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


#
# class Actor_Critic_Transformer(nn.Module):
#     def __init__(self, args):
#         super(Actor_Critic_Transformer, self).__init__()
#         self.use_gru = args.use_gru
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
#
#         self.actor_rnn_hidden = None
#         self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
#         if args.use_gru:
#             print("------use GRU------")
#             self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
#         else:
#             print("------use LSTM------")
#             self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
#
#         self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)
#
#         self.critic_rnn_hidden = None
#         self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
#         if args.use_gru:
#             self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
#         else:
#             self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
#         self.critic_fc2 = nn.Linear(args.hidden_dim, 1)
#
#         if args.use_orthogonal_init:
#             print("------use orthogonal init------")
#             orthogonal_init(self.actor_fc1)
#             orthogonal_init(self.actor_rnn)
#             orthogonal_init(self.actor_fc2, gain=0.01)
#             orthogonal_init(self.critic_fc1)
#             orthogonal_init(self.critic_rnn)
#             orthogonal_init(self.critic_fc2)
#
#     def actor(self, s):
#         s = self.activate_func(self.actor_fc1(s))
#         output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
#         logit = self.actor_fc2(output)
#         return logit
#
#     def critic(self, s):
#         s = self.activate_func(self.critic_fc1(s))
#         output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
#         value = self.critic_fc2(output)
#         return value


class PPO_discrete_Transformer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # Learning rate of actor
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # self.ac = Actor_Critic_Transformer(args)

        self.actor = Actor_Transformer(args)
        self.critic = Critic_Transformer(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
            self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        else:
            pass
            # self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    # def reset_rnn_hidden(self):
    #     self.ac.actor_rnn_hidden = None
    #     self.ac.critic_rnn_hidden = None

    def choose_action(self, s, evaluate=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            # Get output from the last observation
            logit = self.actor(s)[:, -1]
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            value = self.critic(s)[:, -1]
            return value.item()

    def train(self, replay_buffer, total_steps, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

        batch = replay_buffer.get_training_data(device=device)  # Get training data

        # Optimize policy for K epochs:
        actor_losses = []
        critic_losses = []

        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                # self.reset_rnn_hidden()
                logits_now = self.actor(
                    batch['s'][index])  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.critic(batch['s'][index]).squeeze(
                    -1)  # values_now.shape=(mini_batch_size, max_episode_len)

                dist_now = Categorical(logits=logits_now)
                dist_entropy = dist_now.entropy()  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1,
                                        surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # critic_loss
                critic_loss = (values_now - batch['v_target'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

                # Update
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()

                actor_loss.backward()
                (critic_loss * 0.5).backward()
                # loss = actor_loss + critic_loss * 0.5
                # loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optim_actor.step()
                self.optim_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        return np.mean(actor_losses), np.mean(critic_losses)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optim_actor.param_groups:
            p['lr'] = lr_now
        for p in self.optim_critic.param_groups:
            p['lr'] = lr_now

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(),
                   "./model/PPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed,
                                                                                    int(total_steps / 1000)))
        torch.save(self.critic.state_dict(),
                   "./model/PPO_critic_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed,
                                                                                     int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(
            torch.load("./model/PPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
        self.critic.load_state_dict(
            torch.load("./model/PPO_critic_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))

import os
import torch
import numpy as np

from nn.critic import FF_V, LSTM_V, GRU_V
from nn.actor import FF_Stochastic_Actor, LSTM_Stochastic_Actor, GRU_Stochastic_Actor

#NOTE: This should go elsewhere for better module organization?
# For interactive eval
import sys
import tty
import termios
import select

class RecursiveSystem:
    """
    A class which implements a stack (as opposed to tree) of policies, critics, and envs called recursively for hierarchical RL.
    Better name probably needed
    """
    def __init__(self, policy_stack, critic_stack, env_fn_stack, deterministic_stack):
        self.policy_stack = policy_stack
        self.critic_stack = critic_stack
        self.env_fn_stack = env_fn_stack
        self.env_head     = env_fn_stack[0](env_fn_stack[1:])
        subenv = self.env_head
        self.deterministic_stack = deterministic_stack
        self.eval_deterministic_stack = [True for _ in range(len(self.policy_stack))]
        #prod = 1
        #while not subenv.has_pd_env:
        #    #print("{} prod {} * {}".format(subenv.__class__, prod, subenv.planner_rate))
        #    prod *= subenv.planner_rate
        #    subenv = subenv.subenv
        #self.rate = prod

    def reset(self, *args, **kwargs):
        for (actor, critic) in zip(self.policy_stack, self.critic_stack):
            if hasattr(actor, 'init_hidden_state'):
                actor.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

        self.top_state = self.env_head.reset()

    def step(self, *args, deterministic=True, render=False, update_norm=False, mirror=False, keypress=None, **kwargs):

        # if evaluating then overrite deterministic_stack
        if deterministic:
            deterministic_stack = self.eval_deterministic_stack
        else:
            deterministic_stack = self.deterministic_stack

        with torch.no_grad():
            if mirror:
                state = self.env_head.mirror_state(self.top_state)
            else:
                state = self.top_state

            # print("depth: {}\t det: {}".format(0, deterministic_stack[0]))

            top_action = self.policy_stack[0](state, deterministic=deterministic_stack[0], update_norm=update_norm).numpy()

            if mirror:
                top_action = self.env_head.mirror_action(top_action)

            recursive_sample = self.env_head.step(top_action,
                                                  self.policy_stack[1:],
                                                  deterministic=deterministic_stack[1:],
                                                  update_norm=update_norm,
                                                  render=render,
                                                  mirror=mirror,
                                                  keypress=keypress)
                                                  #calling_rate=self.rate)

            #if not hasattr(rewards, 'shape') or len(rewards.shape) < 1: # top-level recursive returns are not batched (n x m), so batch em (1 x n x m)
            #    states  = np.expand_dims(states, 0)
            #    actions = np.expand_dims(actions, 0)
            #    rewards = np.expand_dims(rewards, 0)
            #    dones   = np.expand_dims(dones, 0)

            if len(recursive_sample) != 5: # special case for no planner stacked on normal env
                next_state, reward, done, subsample = recursive_sample
            else:
                next_state, top_action, reward, done, subsample = recursive_sample
                #ret = (self.top_state, top_action) + recursive_sample[2:]

            ret = (np.expand_dims(self.top_state, 0), np.expand_dims(top_action, 0), np.expand_dims(np.array([reward]), 0), np.expand_dims(np.array(done), 0), subsample)
            #print("{} depth {} subsamples len: {}".format('', 0, len(subsample)))

            #wtf = ret
            #d = 1
            #while wtf is not None and len(wtf) == 5:
            #    print('depth {} subsample (depth {}) len {}'.format(d, d+1, len(wtf)))
            #    wtf = wtf[4]
            #    d += 1
            #print("no more subsamples!")

            #print("TOP LEVEL PLANNER {} RETURNING {}, {}, {},".format(self.env_head.__class__, self.top_state.shape, top_action.shape, recursive_sample[2], recursive_sample[3]))
            self.top_state = next_state
            #self.top_state = recursive_sample[0]
            #print('\n')
            return ret

    def save(self):
        pass

    def eval(self, trials, max_len, *args, deterministic=True, render=False, verbose=True, update_norm=False, mirror=False, **kwargs):
        evals = np.zeros(len(self.policy_stack))
        for _ in range(trials):
            steps = 0
            done = False
            self.reset()
            while not done and steps < max_len:
                recursive_sample = self.step(deterministic=deterministic, render=render, update_norm=update_norm, mirror=mirror)
                lens = []
                for depth in range(len(self.policy_stack)):
                    #print("recursive sample at depth {} is: {}".format(depth+1, recursive_sample))
                    states, actions, rewards, dones, next_level = recursive_sample
                    evals[depth] += np.sum(rewards)
                    level_done = done or dones if isinstance(dones, bool) else True in dones
                    if level_done:
                        done = True
                    recursive_sample = next_level
                    lens.append(1 if depth == 0 else len(states))
                steps += max(lens)
        return evals / trials

class Learning_Tree():
    """
    A class which implements a tree of learning policies for hierarchical RL.
    """
    def __init__(self, tree_info):

        self.name = tree_info["name"]
        self.env = tree_info["env"]
        
        obs_dim = tree_info["obs_dim"]
        action_dim = tree_info["action_dim"]
        std = torch.ones(action_dim) * tree_info["std"]
        bounded = tree_info["bounded"]
        layers = tree_info["layers"]
        
        if tree_info["arch"].lower() == 'lstm':
            self.actor = LSTM_Stochastic_Actor(obs_dim, action_dim, env_name=self.env, fixed_std=std, bounded=False, layers=layers)
            self.critic = LSTM_V(obs_dim, layers=layers)
        elif tree_info["arch"].lower() == 'gru':
            self.actor = GRU_Stochastic_Actor(obs_dim, action_dim, env_name=self.env, fixed_std=std, bounded=False, layers=layers)
            self.critic = GRU_V(obs_dim, layers=layers)
        elif tree_info["arch"].lower() == 'ff':
            self.actor = FF_Stochastic_Actor(obs_dim, action_dim, env_name=self.env, fixed_std=std, bounded=False, layers=layers)
            self.critic = FF_V(obs_dim, layers=layers)
        else:
            raise RuntimeError
        
        if tree_info["child_policy"] is not None:
            self.child_policy = Learning_Tree(tree_info["child_policy"])
        else:
            self.child_policy = None

        ## References to each of the child networks in current branch of the tree. Useful for applying common functions to actor and critic networks.
        
        # Lists Ordered from root node to leaf node
        self.policies_list = self._get_all_actors()
        self.critics_list = self._get_all_critics()
        self.names_list = self._get_all_names()

        # Dicts for fast look-up
        self.policies_dict = {n:p for n,p in zip(self.names_list, self.policies_list)}
        self.critics_dict = {n:c for n,c in zip(self.names_list, self.critics_list)}
        
        self.is_recurrent = self._check_recurrent()

    def _get_all_names(self, names=[]):
        """
        Add to a list of names to each Learning_Trees
        """
        names.insert(0, self.name)
        if self.child_policy is not None:
            return self.child_policy.names_list
        return names

    def _get_all_actors(self, actors=[]):
        """
        Add to a list of references to actors
        """
        actors.insert(0, self.actor)
        if self.child_policy is not None:
            return self.child_policy.policies_list
        return actors
        
    def _get_all_critics(self, critics=[]):
        """
        Add to a list of references to critics
        """
        critics.insert(0, self.critic)
        if self.child_policy is not None:
            return self.child_policy.critics_list
        return critics

    def _check_recurrent(self):
        """
        Check if any networks in the model tree are recurrent
        """
        recurrent = False
        for p in self.policies_list:
            if p.is_recurrent:
                recurrent = True
                break
        return recurrent

    def init_hidden_state(self):
        for a in self.policies_list:
            if hasattr(a, 'init_hidden_state'):
                a.init_hidden_state()
        for c in self.critics_list:
            if hasattr(a, 'init_hidden_state'):
                a.init_hidden_state()

    def noisy_forward(self, x, noise, deterministic=True, update_norm=False):
        """
        Return final action with added noise for each level of hierarchy only. Used for input normalization
        """
        for a in self.policies_list:
            x = a.forward(x, deterministic=deterministic, update_norm=update_norm)
            x += torch.randn(a.action_dim) * (noise**0.5)
        return x

    def forward(self, x, deterministic=True, update_norm=False):
        """
        Return final action only. Used for evaluation
        """
        for a in self.policies_list:
            x = a.forward(x, deterministic=deterministic, update_norm=update_norm)
        return x

    # TODO: rewrite this using new self.policies_list list for clarity
    def actors_forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        """
        Return all actor actions including final action as expressive dict
        """
        ret = {}
        for i,a in enumerate(self.policies_list):
            action = a.forward(x, deterministic=deterministic, update_norm=update_norm, return_log_probs=return_log_probs)
            ret[self.names_list[i]] = action
            x = action
        return ret

    # TODO: rewrite this using new self.critics_list list for clarity
    def critics_forward(self, x):
        """
        Return all critic values including final values as expressive dict
        """
        ret = {}
        for i,c in enumerate(self.critics_list):
            value = c.forward(x)
            ret[self.names_list[i]] = value
            x = value
        return ret

    def __call__(self, x, deterministic=True, update_norm=False):
        """
        Another way to call forward(x)
        """
        return self.forward(x, deterministic=deterministic, update_norm=update_norm)

def save_learning_tree(path, learning_tree, name):
    # Given a base experiment path save the learning tree object to it
    actor_path = os.path.join(path, "actor_{}.pt".format(name))
    critic_path = os.path.join(path, "critic_{}.pt".format(name))
    # linear search for policy and critic
    for i,n in enumerate(learning_tree.names_list):
        if n == name:
            torch.save(learning_tree.policies_list[i], actor_path)
            torch.save(learning_tree.critics_list[i], critic_path)
            return

# TODO @yeshg: this shouldn't be too much additional work to write.
def load_learning_tree(oath):
    # Given a base experiment path load the learning tree object from it
    raise NotImplementedError


# planner_controller_tree_info = {
#     "name" : "planner",
#     "arch" : "lstm",
#     "obs_dim" : x + state_x,
#     "action_dim" : y,
#     "env_name" : "cassie_planner",
#     "bounded" : False,
#     "fixed_std" : 0.13,
#     "child_policy" : {
#         "name" : "controller",
#         "arch" : "lstm",
#         "obs_dim" : y + state_y,
#         "action_dim" : 6,
#         "env_name" : "cassie",
#         "bounded" : False,
#         "fixed_std" : 0.13,
#         "child_policy" : {
#             "name" : "taskspace_transform",
#             "arch" : "ff",
#             "obs_dim" : 6,
#             "action_dim" : 10,
#             "env_name" : "cassie"
#             "bounded" : False
#             "fixed_std" : 0.13,
#             "child_policy" : None
#         }
#     }
# }

# controller_tree_info = {
#     "name" : "controller",
#     "arch" : "lstm",
#     "in_dim" : 40,
#     "action_dim" : 6,
#     "env_name" : "cassie",
#     "bounded" : False,
#     "fixed_std" : 0.13,
#     "child_policy" : {
#         "name" : "taskspace_transform",
#         "arch" : "ff",
#         "in_dim" : 6,
#         "action_dim" : 10,
#         "env_name" : "cassie"
#         "bounded" : False
#         "fixed_std" : 0.13,
#         "child_policy" : None
#     }
# }

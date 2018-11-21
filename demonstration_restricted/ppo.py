import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import normal
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch.optim as optim

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, n_inp, n_out, hidden, init_w = None):
        super(Actor, self).__init__()
        if init_w is None : init_w = 1e-3
        self.num_hidden_layers = hidden
        
        self.input = n_inp
        self.output_action = n_out

        #Dense Block
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)        
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)        
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)

    def init_weights(self, init_w):
        self.dense_1.weight.data = fanin_init(self.dense_1.weight.data.size())
        self.dense_2.weight.data = fanin_init(self.dense_2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

    def forward(self, input):
        x = torch.from_numpy(input).type(torch.cuda.FloatTensor)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        mu= F.tanh(self.output(x))
        std = F.softplus(self.output(x))
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_inp, n_out, hidden):
        super(Critic, self).__init__()
        self.n_inp = n_inp
        self.n_out = n_out
        self.dense_1 = nn.Linear(self.n_inp, hidden)
        self.dense_2 = nn.Linear(hidden, hidden)
        self.dense_3 = nn.Linear(hidden, 1)

    def forward(self, input):
        x = F.relu(self.dense_1(input))
        x = F.relu(self.dense_2(x))
        value = self.dense_3(x)
        return value

def value_loss(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma):
    states = np.vstack(np.array(states))
    actions = np.vstack(np.array(actions))
    next_states = np.vstack(np.array(next_state))
    rewards = np.vstack(np.array(rewards))
    next_state = next_state.reshape((1,-1))
    if done:
        reward_sum = 0.
    else:
        reward_sum = critic_target(torch.from_numpy(next_state).type(torch.cuda.FloatTensor))
    
    discounted_rewards = []
    for reward in rewards[::-1]:
        reward_sum = torch.from_numpy(reward).type(torch.cuda.FloatTensor) + gamma*reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    
    values = critic(torch.from_numpy(states).type(torch.cuda.FloatTensor))
    advantage = torch.stack((discounted_rewards)) - values
    value_loss = (advantage**2)/2
    value_loss = torch.mean(value_loss)
    return value_loss

def ppo_iter(mini_batch_size, states, actions, probs_acts, rewards, next_states):
    states = np.vstack(np.array(states))
    actions = np.vstack(np.array(actions))
    probs_acts = np.vstack(np.array(probs_acts))
    next_states = np.vstack(np.array(next_states))
    rewards = np.vstack(np.array(rewards))
    batch_size = states.shape[0]
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], probs_acts[rand_ids, :], rewards[rand_ids, :], next_states[rand_ids, :]

def actor_update(epochs, mini_batch_size, states, actions, probs_acts, rewards, next_states, actor, critic, critic_target, gamma, ent_level):
    clip_p = 0.2
    
    act_losses = []
    for state_b, action_b, probs_b, rews_b, next_state_b in ppo_iter(mini_batch_size, states, actions, probs_acts, rewards, next_states):
        for _ in range(epochs): 
            new_probs_b = []
            entropies = []
            ratio_b = []
            for st, ac, p in zip(state_b, action_b, probs_b):
                mu, sigma = actor(st.reshape((1,-1)))
                dist = normal.Normal(mu, sigma)
                entropies.append(dist.entropy())
                new_prob = torch.exp(dist.log_prob(torch.from_numpy(ac).type(torch.cuda.FloatTensor)))
                p+=1e-10
                ratio_b.append(torch.prod(new_prob/torch.from_numpy(p).type(torch.cuda.FloatTensor)))
                new_probs_b.append(new_prob)
            entropies = torch.sum(torch.stack((entropies)))
            new_probs_b = torch.reshape(torch.stack((new_probs_b)), shape = (-1,1))

            advantage_b = torch.from_numpy(rews_b).type(torch.cuda.FloatTensor) + gamma*critic_target(torch.from_numpy(next_state_b).type(torch.cuda.FloatTensor)) - critic_target(torch.from_numpy(state_b).type(torch.cuda.FloatTensor))
            advantage_b = torch.clamp(advantage_b, -1, 1)
            
            ratio_b = torch.reshape(torch.stack(ratio_b), shape = (-1,1))
            surr1 = ratio_b*(advantage_b).detach()
            clipped = torch.clamp(ratio_b, 1.0 - clip_p, 1.0 + clip_p)
            surr2 = clipped*(advantage_b).detach()
            act_loss = torch.min(surr1, surr2)
            act_loss = -torch.mean(act_loss) - entropies*ent_level
            act_losses.append(act_loss)
            opt= optim.Adam(actor.parameters(), lr= 0.0001)
            opt.zero_grad()
            act_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            opt.step()
            act_losses_sum = torch.mean(torch.stack(act_losses))
            #if want_grads : return act_losses_sum, grads
            return act_losses_sum

def critic_update(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma):
    tau = 0.01

    loss = value_loss(states, actions, next_states, rewards, next_state, done, critic, critic_target, gamma)
    opt= optim.Adam(critic.parameters(), lr= 0.001)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
    opt.step()
    for param, shared_param in zip(critic.parameters(),
                                   critic_target.parameters()):
        shared_param._grad = param.grad*tau + (1-tau)*shared_param
    return loss






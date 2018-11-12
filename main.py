from ppo import *
from dyn_model import *
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import gym
import torch.optim as optim
import torch.nn as nn
from torch.distributions import normal

plot = plt.plot
env = gym.make("FetchPickAndPlace-v1")
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

states, actions, next_states = [], [], []

#print (env.action_space.sample())
for ep in range(10):
    state = env.reset()
    for st in range(20):
        states.append(state)
        act = env.action_space.sample()
        next_state, _, _, _ = env.step(act)
        actions.append(act)
        next_states.append(next_state)
        state = next_state
#env.observation_space = torch.Tensor(env.observation_space)
#env.action_space = torch.Tensor(env.action_space)

obs_dim = getattr(env.observation_space.shape[0], "tolist", lambda x= env.observation_space.shape[0]: x)()
act_dim = getattr(env.action_space.shape[0], "tolist", lambda x= env.action_space.shape[0]: x)()
#print (obs_dim, act_dim)
#getattr(value, "tolist", lambda x=value: x)()

normalizer = Normalizer(state_size = obs_dim, act_size = act_dim)
#print (states.size(), actions.size(), next_states.size())
norm_dict = normalizer.fit(np.array(states), np.array(actions), np.array(next_states))

state = states[np.random.randint(0, len(states))]
state_norm = (state - norm_dict["state_mean"])/norm_dict["state_std"]
#print (torch.from_numpy(env.observation_space.shape[0]))
dyn1 = NNDynamicsModel(state_size = obs_dim, act_size = act_dim, hid_size = 60, normalization = norm_dict, batch_size = 32, iterations = 5, learning_rate = 1e-3)
dyn1.fit({"states": np.array(states[:len(states)]), "acts": np.array(actions[:len(actions)]), "next_states" : np.array(next_states[:len(next_states)])}, plot = 1)

#act_opt = optim.Adam(lr=0.0001)
#crt_opt = optim.Adam(lr=0.001)

state_dim = obs_dim
action_dim = act_dim

actor = Actor(state_dim, action_dim, 64, init_w= .01)
critic = Critic(state_dim, action_dim, 64)
critic_target = Critic(state_dim, action_dim, 64)

states = []
actions = []
next_states = []
rewards = []
probs_acts = []
v_losses = []
a_losses= []
gamma = 0.99
noise_factor = 1.
r_list = []
ent_level = 0.01
stop = 1000

def compute_intr_reward(state, act, next_state):
    pred_dyn = dyn1.predict(state, act, next_states = True)
    #print (next_state.size(), pred_dyn.size())
    criterion = nn.MSELoss()
    return criterion(torch.from_numpy(pred_dyn), torch.from_numpy(np.expand_dims(next_state, axis = 0))).numpy()*1e2

done = False
new_states, new_actions, new_next_states = [], [], []

r_avg = 0.
PRINT = False
rand_w = .8
ep_rew_avg = 0
ev_rews = []
ep_rew_mean = 0
max_eps = 10000
max_steps = 100
ep_numb = 0
ep_avgs_list = []
while ep_numb < max_eps:
    states = []
    actions = []
    next_states = []
    rewards = []
    probs_acts = []
    ep_numb+=1
    state = env.reset()
    episode_reward = 0
    state = (state - norm_dict["state_mean"])/(norm_dict["state_std"])
    ep_buffer = []
    for step in range(max_steps):
        mu, sigma = actor(state.reshape((1,-1)))
        dist = normal.Normal(mu, sigma)
        action = dist.sample()
        act_clip = np.clip(action.detach().numpy(), -1, 1)
        act_prob = dist.log_prob(action)
        act_prob = np.exp(act_prob.detach().numpy()).reshape((1,-1))
        next_state, reward, done, _ = env.step(action.numpy()[0])
        done = False
        state_nat = state*norm_dict["state_std"] + norm_dict["state_mean"]
        reward = compute_intr_reward(state_nat, act_clip, next_state)
        reward = np.clip(reward, -1, 1)
        if step == max_steps-1 : 
            done = True
        next_state = np.array(next_state, dtype = np.float64)
        next_state = (next_state - norm_dict["state_mean"])/(norm_dict["state_std"])
        #env.render()
        episode_reward+=reward

        states.append(state.reshape((1,-1)))
        actions.append(action.numpy().reshape((1,-1)))
        next_states.append(next_state.reshape((1,-1)))
        rewards.append(np.array(reward).reshape((1,-1)))
        probs_acts.append(act_prob)
        
        new_states.append((state*(norm_dict["state_std"])) + norm_dict["state_mean"])
        new_actions.append((action.detach().numpy()*(norm_dict["act_std"])) + norm_dict["act_mean"])
        new_next_states.append((next_state*(norm_dict["state_std"])) + norm_dict["state_mean"])
        
        ep_buffer.append((state.reshape((1,-1)), action.reshape((1,-1))))

        if step % 49 == 0 and not step == 0:
            v_loss = critic_update(states, actions, next_states,
                        rewards, next_state, done, critic, critic_target, gamma)
            v_losses.append(v_loss)
            a = actor_update(2, 8, states, actions, probs_acts, rewards,
                    next_states, actor, critic, critic_target, gamma, ent_level)
            a_losses.append(a)

        if done:                
            ep_rew_avg = (ep_rew_avg*(ep_numb-1))/(ep_numb)  + episode_reward/(ep_numb)
            ep_avgs_list.append(ep_rew_avg)
            print("ep rew avg", ep_rew_avg)
            print("ep reward = ", episode_reward)
            
            for k in range(8):
                v_loss = critic_update(states, actions, next_states,
                        rewards, next_state, done, critic, critic_target, gamma)
                v_losses.append(v_loss)
            a = actor_update(4, 8, states, actions, probs_acts, rewards,
                    next_states, actor, critic, critic_target, gamma, ent_level)
            a_losses.append(a)
            r_list.append(episode_reward) 
            if ep_numb % 100 == 0 and not ep_numb == 0:
                plt.plot(r_list)
                plt.title("Rewards")
                plt.show()
                plt.plot(v_losses)
                plt.title("Critic losses")
                plt.show()
                plt.plot(a_losses)
                plt.title("Actor losses")
                plt.show()
                plt.plot(ep_avgs_list)
                plt.title("Average rewards")
                plt.show()
            break  

        state = next_state    
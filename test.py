from ppo import *
from dyn_model import *
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym

env = gym.make("FetchPickAndPlace-v1")
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVEPATH = os.getcwd() + '/save/fetch_actor_params.pkl'
SAVEPATH2 = os.getcwd() + '/save/fetch_critic_params.pkl'
SAVEPATH3 = os.getcwd() + '/save/fetch_critic2_params.pkl'
SAVEPATH4 = os.getcwd() + '/save/fetch_dyn1_params.pkl'
states, actions, next_states = [], [], []
for ep in range(100):
    state = env.reset()
    for st in range(200):
        states.append(state)
        act = env.action_space.sample()
        next_state, _, _, _ = env.step(act)
        actions.append(act)
        next_states.append(next_state)
        state = next_state

obs_dim = getattr(env.observation_space.shape[0], "tolist", lambda x= env.observation_space.shape[0]: x)()
act_dim = getattr(env.action_space.shape[0], "tolist", lambda x= env.action_space.shape[0]: x)()
#print (obs_dim, act_dim)

normalizer = Normalizer(state_size = obs_dim, act_size = act_dim)
#print (states.size(), actions.size(), next_states.size())
norm_dict = normalizer.fit(np.array(states), np.array(actions), np.array(next_states))

state = states[np.random.randint(0, len(states))]
state_norm = (state - norm_dict["state_mean"])/norm_dict["state_std"]
#print (torch.from_numpy(env.observation_space.shape[0]))

dyn1 = NNDynamicsModel(state_size = obs_dim, act_size = act_dim, hid_size = 60, normalization = norm_dict, batch_size = 32, iterations = 5, learning_rate = 1e-3, device=device)
state_dim = obs_dim
action_dim = act_dim

actor = Actor(state_dim, action_dim, 64, init_w= .01)
critic = Critic(state_dim, action_dim, 64)
critic_target = Critic(state_dim, action_dim, 64)

if os.path.isfile(SAVEPATH):
    actor.load_state_dict(torch.load(SAVEPATH))
    print ("loading actor")
if os.path.isfile(SAVEPATH2):
    critic.load_state_dict(torch.load(SAVEPATH2))
    print ("critic params")
if os.path.isfile(SAVEPATH3):
    print ("loading critic2")
    critic_target.load_state_dict(torch.load(SAVEPATH3))
if os.path.isfile(SAVEPATH4):
    print ("loading dyn params")
    dyn1.network.load_state_dict(torch.load(SAVEPATH4))

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
while True:
    states = []
    actions = []
    next_states = []
    rewards = []
    probs_acts = []
    ep_numb+=1
    #ep_start_time = time.time()
    state = env.reset()
    episode_reward = 0
    state = (state - norm_dict["state_mean"])/(norm_dict["state_std"])
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
        reward2 = compute_intr_reward(state_nat, act_clip, next_state)
        reward = reward2
        reward = np.clip(reward, -1, 1)
        if step == max_steps-1 : 
            done = True
        next_state = np.array(next_state, dtype = np.float64)
        next_state = (next_state - norm_dict["state_mean"])/(norm_dict["state_std"])
        env.render()
        episode_reward+=reward

        states.append(state.reshape((1,-1)))
        actions.append(action.numpy().reshape((1,-1)))
        next_states.append(next_state.reshape((1,-1)))
        rewards.append(np.array(reward).reshape((1,-1)))
        probs_acts.append(act_prob)
        
        new_states.append((state*(norm_dict["state_std"])) + norm_dict["state_mean"])
        new_actions.append((action.detach().numpy()*(norm_dict["act_std"])) + norm_dict["act_mean"])
        new_next_states.append((next_state*(norm_dict["state_std"])) + norm_dict["state_mean"])
        
        #ep_buffer.append((state.reshape((1,-1)), action.reshape((1,-1))))
        
        if done:                
            ep_rew_avg = (ep_rew_avg*(ep_numb-1))/(ep_numb)  + episode_reward/(ep_numb)
            ep_avgs_list.append(ep_rew_avg)
            print("ep rew avg", ep_rew_avg)
            print("ep reward = ", episode_reward) 
        state = next_state   
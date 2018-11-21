import gym
#import time
import random
import numpy as np
from ppo import *
from dyn_model import *
import random
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import normal
import csv

"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

def main():   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    states, actions, next_states = [], [], []
    SAVEPATH = os.getcwd() + '/save/fetch_actor_params.pkl'
    SAVEPATH2 = os.getcwd() + '/save/fetch_critic_params.pkl'
    SAVEPATH3 = os.getcwd() + '/save/fetch_critic2_params.pkl'
    SAVEPATH4 = os.getcwd() + '/save/fetch_dyn1_params.pkl'

    for ep in range(100):
        state = env2.reset()
        for st in range(200):
            states.append(state)
            act = env2.action_space.sample()
            next_state, _, _, _ = env2.step(act)
            actions.append(act)
            next_states.append(next_state)
            state = next_state
    
    obs_dim = getattr(env2.observation_space.shape[0], "tolist", lambda x= env2.observation_space.shape[0]: x)()
    act_dim = getattr(env2.action_space.shape[0], "tolist", lambda x= env2.action_space.shape[0]: x)()
    normalizer = Normalizer(state_size = obs_dim, act_size = act_dim)
    norm_dict = normalizer.fit(np.array(states), np.array(actions), np.array(next_states))

    dyn1 = NNDynamicsModel(state_size = obs_dim, act_size = act_dim, hid_size = 60, normalization = norm_dict, batch_size = 32, iterations = 5, learning_rate = 1e-3, device= device)
    dyn1.fit({"states": np.array(states[:len(states)]), "acts": np.array(actions[:len(actions)]), "next_states" : np.array(next_states[:len(next_states)])}, plot = 1)
    
    state_dim = obs_dim
    action_dim = act_dim

    actor = Actor(state_dim, 1, 64, init_w= .01)
    actor = actor.to(device)
    critic = Critic(state_dim, 1, 64)
    critic = critic.to(device)
    critic_target = Critic(state_dim, 1, 64)
    critic_target = critic_target.to(device)

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

    savefile = os.getcwd() + '/save/mario_curves.csv'
    title = ['ep_reward','episode number','ep_rew_avg']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)

    def compute_intr_reward(state, act, next_state):
        pred_dyn = dyn1.predict(state, act, next_states = True)
        #print (next_state.size(), pred_dyn.size())
        criterion = nn.MSELoss()
        return criterion(torch.from_numpy(pred_dyn).type(torch.cuda.FloatTensor), torch.from_numpy(np.expand_dims(next_state, axis = 0)).type(torch.cuda.FloatTensor)).cpu().numpy()*1e2

    gamma = 0.99
    ent_level = 0.01
    done = False
    max_eps = 100000    
    #max_steps = 50
    ep_numb = 0

    while ep_numb < max_eps:
        states = []
        actions = []
        next_states = []
        rewards = []
        probs_acts = []
        ep_numb+=1
        episode_reward = 0
        lastObs = env.reset()
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03
        timeStep = 0

        

        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
            action = [0, 0, 0, 0]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6
            timeStep += 1
            action[len(action)-1] = 0.05 #open
            env.render()
            obsDataNew, reward, done, info = env.step(action)
            next_state = env2.observation(obsDataNew)
            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            episode_reward += reward
            if timeStep >= env._max_episode_steps: break

        state = env2.observation(lastObs)
        state = (state - norm_dict["state_mean"])/(norm_dict["state_std"])

        while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:            
            
            action = [0, 0, 0, 0]
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i]*3
            
            mu, sigma = actor(state.reshape((1,-1)))
            dist = normal.Normal(mu, sigma)
            action[3] = dist.sample().cpu().numpy()[0][0]     
            action = np.array([action])       
            act_clip = np.clip(action, -1, 1)
            action = torch.from_numpy(action).type(torch.cuda.FloatTensor)
            act_prob = dist.log_prob(action)
            act_prob = np.exp(act_prob.cpu().detach().numpy()).reshape((1,-1))
            obsDataNew, reward, done, _ = env.step(action.cpu().numpy()[0])
            next_state = env2.observation(obsDataNew)
            done = False
            state_nat = state*norm_dict["state_std"] + norm_dict["state_mean"]
            reward = compute_intr_reward(state_nat, act_clip, next_state)
            reward = np.clip(reward, -1, 1)
            next_state = np.array(next_state)
            next_state = (next_state - norm_dict["state_mean"])/(norm_dict["state_std"])
            env.render()
            episode_reward+=reward
            object_rel_pos = obsDataNew['observation'][6:9]
            objectPos = obsDataNew['observation'][3:6]
            timeStep += 1
            states.append(state.reshape((1,-1)))
            actions.append(action.cpu().numpy().reshape((1,-1)))
            next_states.append(next_state.reshape((1,-1)))
            rewards.append(np.array(reward).reshape((1,-1)))
            probs_acts.append(act_prob)
            state = next_state
            
            if done:  
                if ep_numb % 50==0 and not ep_numb == 0:
                    print("  ep reward {}, episode_number {}".format(episode_reward, ep_numb))  

                    data = [episode_reward, ep_numb]
                    with open(savefile, 'a', newline='') as sfile:
                        writer = csv.writer(sfile)
                        writer.writerows([data])
                critic_update(states, actions, next_states,
                        rewards, next_state, done, critic, critic_target, gamma)
                actor_update(4, 8, states, actions, probs_acts, rewards,
                        next_states, actor, critic, critic_target, gamma, ent_level)

                if ep_numb % 50 == 0 and not ep_numb == 0:
                    torch.save(actor.state_dict(), SAVEPATH)
                    torch.save(critic.state_dict(), SAVEPATH2)
                    torch.save(critic_target.state_dict(), SAVEPATH3)
                    torch.save(dyn1.network.state_dict(), SAVEPATH4)
            if timeStep >= env._max_episode_steps: break                    

        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
            env.render()
            action = [0, 0, 0, 0]
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*6

            action[len(action)-1] = -0.005

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            episode_reward+=reward
            if timeStep >= env._max_episode_steps: break
            '''if step % 49 == 0 and not step == 0:
                critic_update(states, actions, next_states,
                    rewards, next_state, done, critic, critic_target, gamma)
                actor_update(2, 8, states, actions, probs_acts, rewards,
                    next_states, actor, critic, critic_target, gamma, ent_level)'''
        
        while True: #limit the number of timesteps in the episode to a fixed duration
            env.render()
            action = [0, 0, 0, 0]
            action[len(action)-1] = -0.005 # keep the gripper closed

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

            if timeStep >= env._max_episode_steps: break    
                 

if __name__ == "__main__":
    main()
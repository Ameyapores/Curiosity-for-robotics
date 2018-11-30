from model import ActorCritic
import os
import time
from collections import deque
import csv
from scipy.misc import imresize
import numpy as np
import cv2
from itertools import count
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import multivariate_normal

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer=None, select_sample=True):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if args.use_cuda else torch.LongTensor

    env = gym.make(args.env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    obs_dim = getattr(env.observation_space.shape[0], "tolist", lambda x= env.observation_space.shape[0]: x)()
    act_dim = getattr(env.action_space.shape[0], "tolist", lambda x= env.action_space.shape[0]: x)()
    model = ActorCritic(obs_dim, act_dim)

    if args.use_cuda:
        model.cuda()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state= env.reset()
    state = torch.from_numpy(state)

    done = True
    episode_length = 0
    for num_iter in count():
        if rank == 0:

            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path)

        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 64)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 64)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values, log_probs, rewards, entropies = [], [], [], []
        actions, forwards, vec_st1s, inverses = [], [], [], []

        for step in range(args.num_steps):
            episode_length += 1            
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, mu, sigma, (hx, cx) = model((state_inp, (hx, cx)), False)
            s_t= state
            #print (mu, sigma)
            b = np.zeros((4, 4))
            np.fill_diagonal(b, sigma.cpu().detach().numpy())
            dist = multivariate_normal.MultivariateNormal(mu, torch.from_numpy(b).type(FloatTensor))
            action = dist.sample()
            entropy= dist.entropy()
            log_prob = dist.log_prob(action)
            
            entropies.append(entropy)
            a_t = action.type(FloatTensor)
            action_out = action.to(torch.device("cpu"))
            #log_prob = log_prob.gather(-1, Variable(action))

            state, reward, done, _ = env.step(action_out.numpy()[0][0])
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            state = torch.from_numpy(state)
            s_t1= state

            vec_st1, inverse, forward = model((Variable(s_t.unsqueeze(0)).type(FloatTensor), Variable(s_t1.unsqueeze(0)).type(FloatTensor), a_t), True)
            reward_intrinsic = args.eta* ((vec_st1 - forward).pow(2)).sum(1) / 2.
            reward_intrinsic = reward_intrinsic.to(torch.device("cpu"))
            #print ("reward_intrinsic", reward_intrinsic)
            reward += reward_intrinsic.detach().numpy()
            #print('total_reward', reward)
            
            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = torch.from_numpy(env.reset())

            values.append(value)
            log_probs.append(log_prob)
            reward= torch.from_numpy(reward).type(FloatTensor)
            rewards.append(reward)
            forwards.append(forward)
            vec_st1s.append(vec_st1)
            inverses.append(inverse)
            actions.append(a_t)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)), False)
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        forward_loss = 0
        inverse_loss = 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)
        #print (rewards)
        #print(log_probs)
        #print(entropies)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]
            
            forward_err = forwards[i] - vec_st1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)

            cross_entropy = - (actions[i] * torch.log(inverses[i] + 1e-15)).sum(1)            
            inverse_loss = inverse_loss + cross_entropy
        #print ('other loss', (policy_loss + args.value_loss_coef * value_loss))
        #print ("policy_loss", policy_loss)
        #print (" value", value_loss)
        optimizer.zero_grad()

        ((1-args.beta) * inverse_loss + args.beta * forward_loss).backward(retain_graph=True)
        (args.lmbda * (policy_loss + 0.5 * value_loss)).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()


def test(rank, args, shared_model, counter):

    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

    env = gym.make(args.env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    obs_dim = getattr(env.observation_space.shape[0], "tolist", lambda x= env.observation_space.shape[0]: x)()
    act_dim = getattr(env.action_space.shape[0], "tolist", lambda x= env.action_space.shape[0]: x)()
    model = ActorCritic(obs_dim, act_dim)
    if args.use_cuda:
        model.cuda()
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    savefile = os.getcwd() + '/save/mario_curves.csv'

    title = ['Time','No. Steps', 'Total Reward', 'Episode Length']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)    

    start_time = time.time()
    
    episode_length = 0
    while True:
        episode_length += 1
        ep_start_time = time.time()
        if done:
            model.load_state_dict(shared_model.state_dict())
            with torch.no_grad():
                cx = Variable(torch.zeros(1, 64)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 64)).type(FloatTensor)

        else:
            with torch.no_grad():
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)
        with torch.no_grad():
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)

        value, mu, sigma, (hx, cx) = model((state_inp, (hx, cx)), False)
        b = np.zeros((4, 4))
        np.fill_diagonal(b, sigma.cpu().detach().numpy())
        dist = multivariate_normal.MultivariateNormal(mu, torch.from_numpy(b).type(FloatTensor))
        action = dist.sample()
        action_out = action.to(torch.device("cpu"))

        state, reward, done, _ = env.step(action_out.numpy()[0][0])
        #env.render()
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)), 
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            
            data = [time.time() - ep_start_time,
                    counter.value, reward_sum, episode_length]
            
            with open(savefile, 'a', newline='') as sfile:
                writer = csv.writer(sfile)
                writer.writerows([data])
            
            reward_sum = 0
            episode_length = 0
            time.sleep(60)
            state = env.reset()
        state = torch.from_numpy(state)



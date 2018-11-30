from model import ActorCritic
import os
import torch
from torch.autograd import Variable
from torch.distributions import multivariate_normal
import numpy as np
import gym

cuda= str(torch.cuda.is_available())

env = gym.make('FetchPickAndPlace-v1')
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

obs_dim = getattr(env.observation_space.shape[0], "tolist", lambda x= env.observation_space.shape[0]: x)()
act_dim = getattr(env.action_space.shape[0], "tolist", lambda x= env.action_space.shape[0]: x)()
model = ActorCritic(obs_dim, act_dim)
if cuda:
    model.cuda()
model.eval()

state = env.reset()
state = torch.from_numpy(state)
reward_sum = 0
done = True
    
episode_length = 0
while True:
    episode_length += 1
    if done:
        model.load_state_dict(torch.load(os.getcwd() + '/save/mario_a3c_params.pkl'))
        with torch.no_grad():
            cx = Variable(torch.zeros(1, 64)).type(torch.cuda.FloatTensor)
            hx = Variable(torch.zeros(1, 64)).type(torch.cuda.FloatTensor)

    else:
        with torch.no_grad():
            cx = Variable(cx.data).type(torch.cuda.FloatTensor)
            hx = Variable(hx.data).type(torch.cuda.FloatTensor)
    with torch.no_grad():
        state_inp = Variable(state.unsqueeze(0)).type(torch.cuda.FloatTensor)

    value, mu, sigma, (hx, cx) = model((state_inp, (hx, cx)), False)
    b = np.zeros((4, 4))
    np.fill_diagonal(b, sigma.cpu().detach().numpy())
    dist = multivariate_normal.MultivariateNormal(mu, torch.from_numpy(b).type(torch.cuda.FloatTensor))
    action = dist.sample()
    action_out = action.to(torch.device("cpu"))

    state, reward, done, _ = env.step(action_out.numpy()[0][0])
    env.render()
    done = done or episode_length >= 50
    reward_sum += reward

    
    if done:
        print("reward", reward_sum)
            
        reward_sum = 0
        episode_length = 0
        state = env.reset()
    state = torch.from_numpy(state)
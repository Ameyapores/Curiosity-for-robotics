import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as optim

class Normalizer():
    def __init__(self, state_size, act_size):
        self.state_size = state_size
        self.act_size = act_size
        self.state_mean = torch.zeros((1, self.state_size))
        self.state_std = torch.zeros((1, self.state_size))
        
        self.action_mean = torch.zeros((1, self.act_size))
        self.action_std = torch.zeros((1, self.act_size))
        
        self.delta_state_mean = torch.zeros((1, self.state_size))
        self.delta_state_std = torch.zeros((1, self.state_size))
        
        self.norm_dict = {"state_mean" : self.state_mean, "state_std" : self.state_std, "act_mean" : self.action_mean, "act_std" : self.action_std, "delta_state_mean" : self.delta_state_mean, "delta_state_std" : self.delta_state_std} 
      
        self.num_samples = 0

    def fit(self, states, actions, next_states):
        #states/acts should be given as (num_samples, state/act)
        #Fit works as a constant moving average: re-fitting data means updating the stats. 
        # It's useful because the distribution of states can change changing the policy over time.
        
        delta_states = next_states - states
        
        self.state_mean[:,:] = torch.mean(states, dim=1)
        self.action_mean[:,:] = torch.mean(actions, dim=1)
        self.delta_state_mean[:,:] = torch.mean(delta_states, dim=1)
        
        self.state_std[:,:] = torch.std(states, dim=1)
        self.action_std[:,:] = torch.std(actions, dim=1)
        self.delta_state_std[:,:] = torch.std(delta_states, dim=1)
        
        new_samples = len(states)
        
        self.norm_dict["state_mean"] = self.norm_dict["state_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.state_mean* (new_samples/(self.num_samples + new_samples))
        self.norm_dict["act_mean"] = self.norm_dict["act_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.action_mean*(new_samples/(self.num_samples + new_samples))
        self.norm_dict["delta_state_mean"] = self.norm_dict["delta_state_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.delta_state_mean*(new_samples/(self.num_samples + new_samples))
        
        self.norm_dict["state_std"] = self.norm_dict["state_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.state_std*(new_samples/(self.num_samples + new_samples)) + 1e-8
        self.norm_dict["act_std"] = self.norm_dict["act_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.action_std*(new_samples/(self.num_samples + new_samples)) + 1e-8
        self.norm_dict["delta_state_std"] = self.norm_dict["delta_state_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.delta_state_std*(new_samples/(self.num_samples + new_samples)) + 1e-8

        
        self.num_samples+=len(states)
        print(self.num_samples)
        
        return self.norm_dict

class DynNet(nn.Module):
    def __init__(self, n_inp, n_out, hid_size):
        super(DynNet, self).__init__()
        self.dense_1 = nn.Linear(self.input, self.num_hidden_layers)
        self.dense_2 = nn.Linear(self.num_hidden_layers, self.num_hidden_layers)
        self.output = nn.Linear(self.num_hidden_layers, self.output_action)
        self.n_inp = n_inp
        self.initialize()

    def initialize(self):
        self.call(torch.randn(1, self.n_inp))
        for var in self.variables:
            if not "bias" in var.name:
                print(var.shape)
    
    def forward(self, input):
        x = F.relu(self.dense_1(input))
        x = F.relu(self.dense_2(x))
        output = F.softmax(self.output(x))
        return output

class NNDynamicsModel():
    def __init__(self, 
                 state_size, 
                 act_size,
                 hid_size,  
                 normalization, #dict that has mean and std for each feature of state, acts, delta_states
                 batch_size,
                 iterations,
                 learning_rate               
                 ):
        self.state_size = state_size
        self.act_size = act_size

        self.network = DynNet(n_inp = self.state_size + self.act_size, n_out = self.state_size, hid_size = hid_size) 

        self.norm_dict = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(lr=learning_rate)

    def train_step(self, x, y, loss_mean):
        out = self.network(x) 
        loss = nn.MSELoss(y, out)
        if loss <  100000:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            max_grad_norm= 50
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
            self.optimizer.step()
            return loss, True
        return 0, False

    def fit(self, trajs, plot = 0):

        states, acts, next_states = trajs["states"], trajs["acts"], trajs["next_states"]
        delta_states = (next_states - states)
        print("states shape", states.shape)
        print("acts shape", acts.shape)
        print("delta states shape", delta_states.shape)

        state_mean = self.norm_dict["state_mean"]
        state_std = self.norm_dict["state_std"]
        act_mean = self.norm_dict["act_mean"]
        act_std = self.norm_dict["act_std"]
        delta_state_mean = self.norm_dict["delta_state_mean"]
        delta_state_std = self.norm_dict["delta_state_std"]

        print("states mean shape", state_mean.shape, state_std.shape)
        states = (states - state_mean)/state_std
        
        print("acts mean shape", act_mean.shape, act_std.shape)
        acts = (acts - act_mean)/act_std
        
        print("delta_states mean shape", delta_state_mean.shape, delta_state_std.shape)
        delta_states = (delta_states - delta_state_mean)/delta_state_std

        inputs = torch.cat((states, acts), 1)
        train_indices = torch.arange(states.shape[0])

        val_indices = train_indices[-len(train_indices)//10:] #take last 10 percent so you sould always pick the new data
        train_indices = train_indices[: -len(train_indices)//10] #take first 90 percent

        print("train indices shape", train_indices.shape)
        print("eval indices shape", val_indices.shape)

        if plot:
            losses = []
        loss_mean = 0
        l_i = 1
        for ep in range(self.iterations):
            torch.randperm(train_indices)
            for batch in range((len(train_indices) // self.batch_size) + 1): # +1 to get the final batch, smaller than batch_size
                indices_batch = train_indices[start : start + self.batch_size]
                input_batch = inputs[indices_batch, :]
                output_batch = delta_states[indices_batch, :]
                input_batch+=torch.normal(out =input_batch.shape)*0.0005
                if l_i == 1:
                        loss, check = self.train_step(input_batch, output_batch, 100000)
                else:
                    loss, check = self.train_step(input_batch, output_batch, loss_mean)
                if check:
                    loss_mean = loss_mean*(l_i-1)/l_i + loss*(1/l_i)
                    l_i+=1
                if plot: 
                    losses.append(loss)
        if plot:
            plt.plot(np.array(losses))
            plt.show()

        val_loss = []
        for batch in range((len(val_indices) // self.batch_size) + 1): # +1 to get the final batch, smaller than batch_size
            start = batch*self.batch_size
            indices_batch = val_indices[start : start + self.batch_size]

            input_batch = inputs[indices_batch, :]
            output_batch = delta_states[indices_batch, :]
            out = self.network(input_batch)
            loss = nn.MSELoss(output_batch, out)
            val_loss.append(loss)
            print(loss)

        print("Validation average loss", torch.mean(val_loss))

    def predict(self, states, acts, next_states = False):
        norm_states = (states - self.norm_dict["state_mean"])/self.norm_dict["state_std"]
        norm_acts = (acts - self.norm_dict["act_mean"])/self.norm_dict["act_std"]
        inp = torch.cat((norm_states, norm_acts), 1) #concatenate along features axis
        out = self.network(inp)
        out = out*self.norm_dict["delta_state_std"] + self.norm_dict["delta_state_mean"]

        if next_states: out+=states
        return out


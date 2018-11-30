import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class ActorCritic(torch.nn.Module):
    
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.linear1= nn.Linear(num_inputs, 64)
        self.linear2= nn.Linear(64, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.critic_linear= nn.Linear(64, 1)
        self.mu = nn.Linear(64, num_actions)
        self.std = nn.Linear(64, num_actions)

        ################################################################
        self.icm_linear1= nn.Linear(num_inputs, 64)
        self.icm_linear2= nn.Linear(64, 64)

        #self.icm_lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.inverse_linear1 = nn.Linear(128, 64)
        self.inverse_linear2 = nn.Linear(64, num_actions)

        self.forward_linear1 = nn.Linear(64 + num_actions, 64)
        self.forward_linear2 = nn.Linear(64, 64)
        
        
        self.apply(weights_init)
        self.linear1.weight.data = normalized_columns_initializer(
            self.linear1.weight.data, 0.01)
        self.linear1.bias.data.fill_(0)
        self.linear2.weight.data = normalized_columns_initializer(
            self.linear2.weight.data, 0.01)
        self.linear2.bias.data.fill_(0)
        self.mu.weight.data = normalized_columns_initializer(
            self.mu.weight.data, 0.01)
        self.linear1.bias.data.fill_(0)
        self.std.weight.data = normalized_columns_initializer(
            self.std.weight.data, 0.01)
        self.std.bias.data.fill_(0)
        self.icm_linear1.weight.data = normalized_columns_initializer(
            self.icm_linear1.weight.data, 0.01)
        self.icm_linear1.bias.data.fill_(0)
        self.icm_linear2.weight.data = normalized_columns_initializer(
            self.icm_linear2.weight.data, 0.01)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.inverse_linear1.weight.data = normalized_columns_initializer(
            self.inverse_linear1.weight.data, 0.01)
        self.inverse_linear1.bias.data.fill_(0)
        self.inverse_linear2.weight.data = normalized_columns_initializer(
            self.inverse_linear2.weight.data, 1.0)
        self.inverse_linear2.bias.data.fill_(0)
        
        self.forward_linear1.weight.data = normalized_columns_initializer(
            self.forward_linear1.weight.data, 0.01)
        self.forward_linear1.bias.data.fill_(0)
        self.forward_linear2.weight.data = normalized_columns_initializer(
            self.forward_linear2.weight.data, 1.0)
        self.forward_linear2.bias.data.fill_(0)
        self.train

    def forward(self, inputs, icm):

        if icm == False:
            """A3C"""
            inputs, (a3c_hx, a3c_cx) = inputs

            x = F.elu(self.linear1(inputs))
            x = F.elu(self.linear2(x))

            #x = x.view(-1, 32 * 3 * 3)
            a3c_hx, a3c_cx = self.lstm(x.view(-1, 64), (a3c_hx, a3c_cx))
            x = a3c_hx

            critic = self.critic_linear(x)
            mu = F.tanh(self.mu(x))
            std = F.softplus(self.std(x))+1e-5
            return critic, mu, std, (a3c_hx, a3c_cx)

        else:
            """icm"""
            s_t, s_t1, a_t = inputs
            '''
            s_t, (icm_hx, icm_cx) = s_t
            s_t1, (icm_hx1, icm_cx1) = s_t1
            '''
            vec_st = F.elu(self.icm_linear1(s_t))
            vec_st = F.elu(self.icm_linear2(vec_st))

            vec_st1 = F.elu(self.icm_linear1(s_t1))
            vec_st1 = F.elu(self.icm_linear2(vec_st1))

            vec_st = vec_st.view(-1, 64)
            vec_st1 = vec_st1.view(-1, 64)

            inverse_vec = torch.cat((vec_st, vec_st1), 1)
            forward_vec = torch.cat((vec_st, a_t), 1)

            inverse = self.inverse_linear1(inverse_vec)
            inverse = F.relu(inverse)
            inverse = self.inverse_linear2(inverse)
            inverse = F.softmax(inverse, dim=0)####

            forward = self.forward_linear1(forward_vec)
            forward = F.relu(forward)
            forward = self.forward_linear2(forward)

            return vec_st1, inverse, forward
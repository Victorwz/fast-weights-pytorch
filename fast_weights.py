from __future__ import print_function

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import time
from retrieval import read_data
from util import Checkpointer

ar_data = read_data()

STEP_NUM = 11
ELEM_NUM = 26 + 10 + 1
HIDDEN_NUM = 20

def softmax_cross_entropy_with_logits(logits, labels):
    loss = torch.sum(-labels * F.log_softmax(logits, -1), -1)
    return loss

class fast_weights_model(nn.Module):
    """docstring for fast_weights_model"""
    def __init__(self, batch_size, step_num, elem_num, hidden_num):
        super(fast_weights_model, self).__init__()
        self.batch_size = batch_size
        self.x = Variable(torch.randn(batch_size, step_num, elem_num).type(torch.float32))
        self.y = Variable(torch.randn(batch_size, elem_num).type(torch.float32))
        self.l = nn.Parameter(torch.tensor([0.9], dtype=torch.float32))
        self.e = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.w1 = nn.Parameter(torch.empty(elem_num, 50).uniform_(-np.sqrt(0.02), np.sqrt(0.02)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros([1, 50]).type(torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.empty(50, 100).uniform_(-np.sqrt(0.01), np.sqrt(0.01)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros([1, 100]).type(torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.empty(hidden_num, 100).uniform_(-np.sqrt(0.01), np.sqrt(0.01)), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros([1, 100]).type(torch.float32), requires_grad=True) 
        self.w4 = nn.Parameter(torch.empty(100, elem_num).uniform_(-np.sqrt(1.0 / elem_num), np.sqrt(1.0 / elem_num)), requires_grad=True)
        self.b4 = nn.Parameter(torch.zeros([1, elem_num]).type(torch.float32), requires_grad=True)

        self.w = nn.Parameter(torch.tensor(0.05 * np.identity(hidden_num)).type(torch.float32), requires_grad=True)

        self.c = nn.Parameter(torch.empty(100, hidden_num).uniform_(-np.sqrt(hidden_num), np.sqrt(hidden_num)), requires_grad=True)

        self.g = nn.Parameter(torch.ones([1, hidden_num]).type(torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.ones([1, hidden_num]).type(torch.float32), requires_grad=True)

    def forward(self, bx, by):
        self.x = torch.tensor(bx)
        self.y = torch.tensor(by) 
        #print(bx.size)
        #print(by.size)
        a = torch.zeros([self.batch_size, HIDDEN_NUM, HIDDEN_NUM]).type(torch.float32)
        h = torch.zeros([self.batch_size, HIDDEN_NUM]).type(torch.float32)

        la = []

        for i in range(0, STEP_NUM):
            s1 = torch.relu(torch.matmul(self.x[:, i, :], self.w1) + self.b1)
            #print(s1.shape, self.w2.shape)
            z = torch.relu(torch.matmul(s1, self.w2) + self.b2)

            h = torch.relu(torch.matmul(h, self.w) + torch.matmul(z, self.c))

            hs = torch.reshape(h, [self.batch_size, 1, HIDDEN_NUM])

            hh = hs

            a = self.l * a + self.e * torch.matmul(hs.transpose(1,2), hs)

            la.append(torch.mean(torch.pow(a,2)))

            for s in range(1):
                hs = torch.reshape(torch.matmul(h, self.w), hh.shape) + \
                     torch.reshape(torch.matmul(z, self.c), hh.shape) + torch.matmul(hs, a)
                mu = torch.mean(hs, 0)
                sig = torch.sqrt(torch.mean(torch.pow((hs - mu), 2), 0))
                hs = torch.relu(torch.div(torch.mul(self.g, (hs - mu)), sig) + self.b)

            h = torch.reshape(hs, [self.batch_size, HIDDEN_NUM])

        h = torch.relu(torch.matmul(h, self.w3) + self.b3)
        logits = torch.matmul(h, self.w4) + self.b4
        correct = torch.argmax(logits, dim=1).eq(torch.argmax(self.y, dim=1))
        self.loss = softmax_cross_entropy_with_logits(logits, self.y).mean()
        self.acc = torch.mean(correct.type(torch.float32))

        return self.loss, self.acc

def train(save = 0, verbose = 0):
    batch_size = cfg.train.batch_size
    model = fast_weights_model(batch_size, STEP_NUM, ELEM_NUM, HIDDEN_NUM)
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.model_lr)
    model.train()
    writer = SummaryWriter(logdir=os.path.join(cfg.logdir, cfg.exp_name), flush_secs=30)
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name))
    start_epoch = 0
    start_epoch = checkpointer.load(model, optimizer)
    batch_idxs = 600
    for epoch in range(start_epoch, cfg.train.max_epochs):
        for idx in range(batch_idxs):
            gloabl_step = epoch * cfg.num_train + idx + 1
            #print(ar_data.train._x)
            bx, by = ar_data.train.next_batch(batch_size=100)
            loss, acc = model(bx, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/loss', loss, gloabl_step)
            writer.add_scalar('acc/acc', acc, gloabl_step)
            if verbose > 0 and idx % verbose == 0:
                print('Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, loss: {:.8f}, acc: {:.2f}'.format(
                    epoch, idx, batch_idxs, time.time() - start_time, loss, acc
                ))
    checkpointer.save(model, optimizer, epoch+1)


if __name__ == "__main__":
    train(verbose = 10)






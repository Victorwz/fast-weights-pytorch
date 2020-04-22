from __future__ import print_function

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
        self.x = Variable(torch.randn(batch_size, step_num, elem_num).type(torch.float32))
        self.y = Variable(torch.randn(batch_size, elem_num).type(torch.float32))
        self.l = torch.tensor([0.9], dtype=torch.float32)
        self.e = torch.tensor([0.5], dtype=torch.float32)

        self.w1 = Variable(torch.empty(elem_num, 50).uniform_(-np.sqrt(0.02), np.sqrt(0.02)))
        self.b1 = Variable(torch.zeros([1, 50]).type(torch.float32))
        self.w2 = Variable(torch.empty(500, 100).uniform_(-np.sqrt(0.01), np.sqrt(0.01)))
        self.b2 = Variable(torch.zeros([1, 100]).type(torch.float32))
        self.w3 = Variable(torch.empty(hidden_num, 100).uniform_(-np.sqrt(0.01), np.sqrt(0.01)))
        self.b3 = Variable(torch.zeros([1, 100]).type(torch.float32)) 
        self.w4 = Variable(torch.empty(100, elem_num).uniform_(-np.sqrt(1.0 / elem_num), np.sqrt(1.0 / elem_num)))
        self.b4 = Variable(torch.zeros([1, elem_num]).type(torch.float32))

        self.w = Variable(torch.tensor(0.05 * np.identity(hidden_num)).type(torch.float32))

        self.c = Variable(torch.empty(100, hidden_num).uniform_(-np.sqrt(hidden_num), np.sqrt(hidden_num)))

        self.g = Variable(torch.ones([1, hidden_num]).type(torch.float32))
        self.b = Variable(torch.ones([1, hidden_num]).type(torch.float32))

    def forward(self, bx, by)
        self.x = bx
        self.y = by
        a = torch.zeros([batch_size, hidden_num, hidden_num]).type(torch.float32)
        h = torch.zeros([batch_size, hidden_num]).type(torch.float32)

        la = []

        for i in range(0, step_num):
            s1 = torch.relu(torch.matmul(self.x[:, t, :], self.w1) + self.b1)
            z = torch.relu(torch.matmul(s1, self.w2) + self.b2)

            h = torch.relu(torch.matmul(h, self.w) + torch.matmul(z, self.c))

            hs = torch.reshape(h, [batch_size, 1, hidden_num])

            hh = hs

            a = self.l * a + self.e * torch.matmul(hs.transpose(1,2), hs)

            la.append(torch.mean(torch.pow(a,2)))

            for s in range(1):
                hs = torch.reshape(torch.matmul(h, self.w), hh.shape) + \
                     torch.reshape(torch.matmul(z, self.c), hh.shape) + torch.matmul(hs, a)
                mu = torch.mean(hs, 0)
                sig = torch.sqrt(torch.mean(torch.pow((hs - mu), 2), 0))
                hs = torch.relu(torch.div(torch.mul(self.g, (hs - mu)), sig) + self.b)

            h = torch.reshape(hs, [batch_size, hidden_num])

        h = torch.relu(torch.matmul(h, self.w3) + self.b3)
        logits = torch.matmul(h, self.w4) + self.b4
        correct = torch.argmax(logits, dim=1).eq(torch.argmax(self.y, dim=1))
        self.loss = softmax_cross_entropy_with_logits(logits, self.y).mean()
        self.acc = torch.mean(correct.type(torch.float32))

        return self.loss, self.acc

def train(self, save = 0, verbose = 0):
    model = fast_weights_model(STEP_NUM, ELEM_NUM, HIDDEN_NUM)
    model.train()
    batch_size = cfg.train.batch_size
    start_time = time.time()
    optimizer = torch.optim.Adam(model.paramters(), lr=cfg.train.model_lr)
    writer = SummaryWriter(logdir=os.path.join(cfg.logdir, cfg.exp_name), flush_secs=30)
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name))
    start_epoch = 0
    start_epoch = checkpointer.load(model, optimizer)
    batch_idxs = 600
    for epoch in range(start_epoch, cfg.train.max_epochs):
        for idx in range(batch_idxs):
            gloabl_step = epoch * cfg.num_train + idx + 1
            bx, by = ar_data.train.next_batch(batch_size=cfg.batch_size)
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






from __future__ import print_function

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, args):
        super(fast_weights_model, self).__init__()
        self.batch_size = args.batch_size
        # Inputs
        self.X = Variable(torch.randn(args.batch_size, args.input_dim, args.num_classes).type(torch.float32))
        # Targets
        self.y = Variable(torch.randn(args.batch_size, args.num_classes).type(torch.float32))
        # Learning Rate
        self.l = torch.tensor([args.learning_rate], dtype=torch.float32)
        # Decay Rate
        self.e = torch.tensor([args.decay_rate], dtype=torch.float32)

        # Input Weights 
        self.W_x = Variable(torch.empty(
                        args.num_classes, 
                        args.hidden_size).uniform_(
                            -np.sqrt(2.0/args.num_classes),
                            np.sqrt(2.0/args.num_classes)
                    ), dtype=torch.float32)
        self.b_x = Variable(torch.zeros(
            [args.hidden_size]
        ), dtype=torch.float32)

        # Hidden weights (initialization explained in Hinton video)
        self.W_h = Variable(initial_value=0.5 * np.identity(args.hidden_size), 
                                                        dtype=torch.float32)

        # Softmax weights
        self.W_softmax = Variable(torch.empty(
                                args.hidden_size, 
                                args.num_classes).uniform_(
                                    -np.sqrt(2.0/args.hidden_size),
                                    np.sqrt(2.0/args.hidden_size)  
                                ), dtype=torch.float32)
        self.b_softmax = Variable(torch.zeros(args.num_classes),
                                            dtype=torch.float32)

        # Scale and shift everything for layernorm
        self.gain = Variable(torch.ones(args.hidden_size), dtype=torch.float32) 
        self.bias = Variable(torch.zeros(args.hidden_size), dtype=torch.float32)


    def forward(self, bx, by):
        self.x = bx
        self.y = by
        a = torch.zeros([self.batch_size, hidden_num, hidden_num]).type(torch.float32)
        h = torch.zeros([self.batch_size, hidden_num]).type(torch.float32)

        la = []

        for i in range(0, step_num):
            s1 = torch.relu(torch.matmul(self.x[:, t, :], self.w1) + self.b1)
            z = torch.relu(torch.matmul(s1, self.w2) + self.b2)

            h = torch.relu(torch.matmul(h, self.w) + torch.matmul(z, self.c))

            hs = torch.reshape(h, [self.batch_size, 1, hidden_num])

            hh = hs

            a = self.l * a + self.e * torch.matmul(hs.transpose(1,2), hs)

            la.append(torch.mean(torch.pow(a,2)))

            for s in range(1):
                hs = torch.reshape(torch.matmul(h, self.w), hh.shape) + \
                     torch.reshape(torch.matmul(z, self.c), hh.shape) + torch.matmul(hs, a)
                mu = torch.mean(hs, 0)
                sig = torch.sqrt(torch.mean(torch.pow((hs - mu), 2), 0))
                hs = torch.relu(torch.div(torch.mul(self.g, (hs - mu)), sig) + self.b)

            h = torch.reshape(hs, [self.batch_size, hidden_num])

        h = torch.relu(torch.matmul(h, self.w3) + self.b3)
        logits = torch.matmul(h, self.w4) + self.b4
        correct = torch.argmax(logits, dim=1).eq(torch.argmax(self.y, dim=1))
        self.loss = softmax_cross_entropy_with_logits(logits, self.y).mean()
        self.acc = torch.mean(correct.type(torch.float32))

        return self.loss, self.acc

def train(save = 0, verbose = 0):
    BATCH_SIZE = 60000
    model = fast_weights_model(BATCH_SIZE, STEP_NUM, ELEM_NUM, HIDDEN_NUM)
    model.train()
    batch_size = cfg.train.batch_size
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.model_lr)
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






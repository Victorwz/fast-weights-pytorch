from collections import defaultdict, deque
import pickle
from attrdict import AttrDict
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)


    def save(self, model, optimizer, epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        filename = os.path.join(self.path, 'model_{:05}.pth'.format(epoch))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if os.path.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)

        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)

    def load(self, model, optimizer):
        """
        Return starting epoch
        """
        with open(self.listfile, 'rb') as f:
            model_list = pickle.load(f)
            if len(model_list) == 0:
                print('No checkpoint found. Starting from scratch')
                return 0
            else:
                checkpoint = torch.load(model_list[-1])
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('Load checkpoint from {}.'.format(model_list[-1]))
                return checkpoint['epoch']

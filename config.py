from attrdict import AttrDict
import os

cfg = AttrDict({
    # 'exp_name': 'test-len10-delta',
    # 'exp_name': 'test-len1-fixedscale-aggre-super',
    # 'exp_name': 'test-aggre-super',
    # 'exp_name': 'test-mask',
    'exp_name': 'test-proposal',
    'resume': True,
    'device': 'cuda:0',
    # 'device': 'cpu',
    
    'train': {
        'batch_size': 100,
        'model_lr': 1e-4,
        'max_epochs': 1000
    },
    'valid': {
        'batch_size': 64
    },
    'num_train': 60000,
    'logdir': 'logs/',
    'checkpointdir': 'checkpoints/',
})
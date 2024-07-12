import os
import argparse
import torch

from torch.backends import cudnn
from controller import Controller


def main(config):
    cudnn.benchmark = True
    
    controller = Controller(vars(config))
    controller.train()
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'memory_initial'])
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--num_workers', type=int, default=4*torch.cuda.device_count())
    
    config = parser.parse_args()
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
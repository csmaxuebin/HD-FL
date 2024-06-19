# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_mnist(num_users):
    # data_train = datasets.MNIST(root="~/data/", train=True, transform=transforms.ToTensor())
    data_train = datasets.MNIST(root="~/data/", train=True, download=True,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor())
    data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    '''
    for i in range(0, 10):
        idx = np.where(data_train.targets == i)
        d = data_train.data[idx].flatten(1).float()
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.flatten(1).float(), data_test.targets.float()))
    '''
    user_dict = mnist_noniid(data_train, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = data_train.data[idx].flatten(1).float()
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.flatten(1).float(), data_test.targets.float()))
    return non_iid





if __name__ == '__main__':
    """
    1. load_data
    2. generate clients (step 3)
    3. generate aggregator
    4. training
    """

    client_num = 10
    d = load_mnist(client_num)


    fl_par = {
        'output_size': 10,
        'client_num': client_num,
        'model': MLP,
        'data': d,
        'lr': 0.001,
        'E': 5,
        'C': 1,
        'epsilon': 1.0,
        'delta': 1e-4,
        'clip': 200,
        'tot_E': 5,
        'batch_size': 32,
        'device': device
    }
    import warnings
    warnings.filterwarnings("ignore")
    fl_entity = FLServer(fl_par).to(device)
    # fl_entity.global_update()
    fl_entity.global_update_grad()





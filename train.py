import torch
import random
import numpy as np

from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torchvision.datasets

from alexnet import AlexNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


# https://github.com/fastai/imagenette
# https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz

X = torch.zeros((1, 3, 224, 224))

device = 'cuda' if torch.cuda.is_available() else 'cpu' # device = 'cpu' # CPU ONLY


net = AlexNet(num_classes=10)
if torch.cuda.is_available():
    net = net.cuda()

if device == 'cuda':
    # make it concurent
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

X = X.to(device)
optimizer.zero_grad()
optputs = net(X)
print(optputs)

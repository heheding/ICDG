import random
import math
import torch.nn as nn
from math import sqrt
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from models.build_gen import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import pyplot as plt
from dataset.dataset_readco import dataset_read
from models.loss import *
from utils import *

seed_all(3407)

def create_model(source_G, P,no_grad=False):
    source_G.load_state_dict(torch.load(model_path+'/g.pth'))
    P.load_state_dict(torch.load(model_path+'/p.pth'))
    if no_grad:
        for param in source_G.parameters():
            param.detach_()
        for param in P.parameters():
            param.detach_()
    return source_G, P

split_s = 0.9
batch_size = 256
optimizer = 'adam'

df = pd.read_csv('/root/dh/2024/OODCo/data/quanbu.csv')
train_s_mean, train_s_std, dataset_s, test_source, s_label_test = dataset_read(
    source = df, split_s = split_s,  batch_size = batch_size)

cuda = True
cudnn.benchmark = True
lr = 5e-4
n_epochs = 70

model_path = '/root/dh/2024/OOD/model_path'

G = Generator(gen='gru')
P = Predictor()

loss_predict = torch.nn.MSELoss(reduction='mean')

lmmd = MMD_loss()
col = CORAL()
device = torch.device("cuda:0")

opt_g = optim.Adam(G.parameters(), lr=lr)
opt_p = optim.Adam(P.parameters(), lr=lr)

# source_G, P = create_model()
G = G.double().to(device)
P = P.double().to(device)
loss_predict = loss_predict.to(device)

for epoch in range(n_epochs):

    G.train()
    P.train()
    for batch_idx, (data_s, label_s) in enumerate(dataset_s):

        data_s = data_s.cuda()
        label_s = label_s.cuda()

        opt_g.zero_grad()
        opt_p.zero_grad()

        feat_s = G(data_s)
        output_s = P(feat_s)
        loss_s = loss_predict(output_s, label_s)
        loss_s.backward()
        opt_g.step()
        opt_p.step()

# test
    
data_t = test_source
label = s_label_test
data_t = data_t.cuda()
feat = G(data_t)
output = P(feat)

pred = output.cpu().detach().numpy()
label = label.detach().numpy()

pred = pred*train_s_std[-1] + train_s_mean[-1]
label = label*train_s_std[-1] + train_s_mean[-1]

# MAE
MAE = mean_absolute_error(pred, label)

# RMSE
RMSE = sqrt(mean_squared_error(pred, label))

# R2得分
R2 = r2_score(pred, label)
print('MAE: {} \t RMSE: {}\t R2: {} \n'.format(MAE, RMSE, R2))
plt.figure(dpi=300,figsize=(28,7))
r = pred.shape[0] + 1
plt.plot(np.arange(1, r), label, 'r-', label="real")
plt.plot(np.arange(1, r), pred, 'g-', label="pred")

plt.legend()
plt.savefig("/root/dh/2024/OOD/preco.png")




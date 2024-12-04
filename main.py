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
from dataset.dataset_read import dataset_read
from models.loss import *
from utils import *

seed_all(3407)

split_s = 0.9
batch_size = 128
optimizer = 'adam'

df = pd.read_csv('/root/dh/2022/91.csv')
train_mean, train_std, dataset1,dataset2,dataset3,dataset4,dataset5,dataset6, test, label_test = dataset_read(
    source = df, split_s = split_s,  batch_size = batch_size)

cuda = True
cudnn.benchmark = True
lr12 = 5e-4
wd12 = 0.0001
n_epochs12 = 90
lr23 = 1e-3
wd23 = 0.0001
n_epochs23 = 100
lr34 = 2.5e-4
wd34 = 0.0001
n_epochs34 = 90
lr45 = 5e-4
wd45 = 0.0001
n_epochs45 = 100
lr56 = 4.9e-4
wd56 = 0.000
n_epochs56 = 150
interval = 10
savemodel= True
loadmodel = True

model_path12 = '/root/dh/2024/OOD/model_path/data12'
model_path23 = '/root/dh/2024/OOD/model_path/data23'
model_path34 = '/root/dh/2024/OOD/model_path/data34'
model_path45 = '/root/dh/2024/OOD/model_path/data45'
model_path56 = '/root/dh/2024/OOD/model_path/data56'

list1 = list(enumerate(dataset1))
list2 = list(enumerate(dataset2))
list4 = list(enumerate(dataset4))
list5 = list(enumerate(dataset5))
list6 = list(enumerate(dataset6))

G12 = Generator(gen='gru12')
P12 = Predictor()
D12 = Discriminator(discri = '')

G23 = Generator(gen='gru')
P23 = Predictor()
D23 = Discriminator(discri = '')

G34 = Generator(gen='gru')
P34 = Predictor()
D34 = Discriminator(discri = '')

G45 = Generator(gen='gru')
P45 = Predictor()
D45 = Discriminator(discri = '')

G56 = Generator(gen='gru')
P56 = Predictor()
D56 = Discriminator(discri = '')

loss_predict = torch.nn.MSELoss(reduction='mean')

lmmd = MMD_loss()
col = CORAL()
device = torch.device("cuda:0")

opt12 = optim.Adam([
    {'params':P12.parameters(), 'lr':lr12, 'weight_decay':wd12}, 
    {'params':G12.parameters(), 'lr':lr12, 'weight_decay':wd12},
    {'params':D12.parameters(), 'lr':lr12, 'weight_decay':wd12}
    ]) 

opt23 = optim.Adam([
    {'params':P23.parameters(), 'lr':lr23, 'weight_decay':wd23}, 
    {'params':G23.parameters(), 'lr':lr23, 'weight_decay':wd23},
    {'params':D23.parameters(), 'lr':lr23, 'weight_decay':wd23}
    ])

opt34 = optim.Adam([
    {'params':P34.parameters(), 'lr':lr34, 'weight_decay':wd34}, 
    {'params':G34.parameters(), 'lr':lr34, 'weight_decay':wd34},
    {'params':D34.parameters(), 'lr':lr34, 'weight_decay':wd34}
    ])  

opt45 = optim.Adam([
    {'params':P45.parameters(), 'lr':lr45, 'weight_decay':wd45}, 
    {'params':G45.parameters(), 'lr':lr45, 'weight_decay':wd45},
    {'params':D45.parameters(), 'lr':lr45, 'weight_decay':wd45}
    ]) 
opt56 = optim.Adam([
    {'params':P56.parameters(), 'lr':lr56, 'weight_decay':wd56}, 
    {'params':G56.parameters(), 'lr':lr56, 'weight_decay':wd56},
    {'params':D56.parameters(), 'lr':lr56, 'weight_decay':wd56}
    ]) 

# source_G, P = create_model()
G12 = G12.double().to(device)
P12 = P12.double().to(device)
D12 = D12.double().to(device)
G23 = G23.double().to(device)
P23 = P23.double().to(device)
D23 = D23.double().to(device)
G34 = G34.double().to(device)
P34 = P34.double().to(device)
D34 = D34.double().to(device)
G45 = G45.double().to(device)
P45 = P45.double().to(device)
D45 = D45.double().to(device)
G56 = G56.double().to(device)
P56 = P56.double().to(device)
D56 = D56.double().to(device)
loss_predict = loss_predict.to(device)

# for epoch in range(n_epochs12):
    
#     start_steps = epoch * len(dataset2)
#     total_steps = n_epochs12 * len(dataset2)
#     batch_j = 0
    
#     # dataset1 and dataset2
#     for batch_idx, (data2, label2) in enumerate(dataset2):
        
#         _, (data1, label1) = list1[batch_j]
#         p = np.double(batch_idx + start_steps) / total_steps
#         constant = 2. / (1. + np.exp(-10 * p)) - 1
#         data1, data2 = data1.cuda(), data2.cuda()
#         label1, label2 = label1.cuda(), label2.cuda()

#         opt12.zero_grad()
        
#         # Recompute feat1 within the loop
#         feat1 = G12(data1, 0)
#         output1 = P12(feat1)
#         feat2 = G12(data2, 0)
#         output2 = P12(feat2)
#         loss1 = loss_predict(output1, label1)
#         loss2 = loss_predict(output2, label2)
#         transfer_loss = D12(feat1, feat2, constant)
#         loss12 = loss1 + loss2 +  1 * transfer_loss
#         loss12.backward()
#         opt12.step()
        
#         batch_j += 1
#         if batch_j >= len(list1):
#             batch_j = 0
        
#     if epoch % interval == 0:
#         print('Epoch12: {}\t Train Epoch: {}\t Leaver: {}\t  loss: {:.6f}\t  loss_D: {:.6f}\t '.format(
#             n_epochs12 ,epoch, '1 and 2',  loss12, transfer_loss))
# if savemodel:
#     ensure_path(model_path12)
#     torch.save(G12.state_dict(), model_path12+'/G12.pth')
#     torch.save(P12.state_dict(), model_path12+'/P12.pth')

# if loadmodel:
#     G12.load_state_dict(torch.load(model_path12+'/G12.pth'))
#     G34.load_state_dict(torch.load(model_path34+'/G34.pth'))
    
# for epoch in range(n_epochs23):
#     start_steps = epoch * len(dataset3)
#     total_steps = n_epochs23 * len(dataset3)
#     batch_j = 0
#     # dataset2 and dataset3
#     for batch_idx, (data3, label3) in enumerate(dataset3):
        
#         _, (data2, label2) = list2[batch_j]
#         p = np.double(batch_idx + start_steps) / total_steps
#         constant = 2. / (1. + np.exp(-10 * p)) - 1
#         data3, data2 = data3.cuda(), data2.cuda()
#         label3, label2 = label3.cuda(), label2.cuda()

#         opt23.zero_grad()
        
#         # Recompute feat1 within the loop
#         with torch.no_grad():
#             feat1 = G12(data3, 0)
#             feat34 = G34(data3, feat1)
#         feat3 = G23(data3, feat34)
#         output3 = P23(feat3)
#         feat2 = G23(data2, feat34)
#         output2 = P23(feat2)
#         loss3 = loss_predict(output3, label3)
#         loss2 = loss_predict(output2, label2)
#         transfer_loss = D23(feat3, feat2, constant)
#         loss23 = loss3 + loss2 +  0.6 * transfer_loss
#         loss23.backward()
#         opt23.step()
        
#         batch_j += 1
#         if batch_j >= len(list2):
#             batch_j = 0
        
#     if epoch % interval == 0:
#         print('Epoch23: {}\t Train Epoch: {}\t Leaver: {}\t  loss: {:.6f}\t  loss_D: {:.6f}\t '.format(
#             n_epochs23 ,epoch, '2 and 3',  loss23, transfer_loss))

# if savemodel:
#     ensure_path(model_path23)
#     torch.save(G23.state_dict(), model_path23+'/G23.pth')
#     torch.save(P23.state_dict(), model_path23+'/P23.pth')

# if loadmodel:
#     G12.load_state_dict(torch.load(model_path12+'/G12.pth'))
#     G23.load_state_dict(torch.load(model_path23+'/G23.pth'))

# for epoch in range(n_epochs34):      
#     start_steps = epoch * len(dataset3)
#     total_steps = n_epochs23 * len(dataset3)  
#     batch_j = 0
#     # dataset3 and dataset4
#     for batch_idx, (data3, label3) in enumerate(dataset3):
        
#         _, (data4, label4) = list4[batch_j]
#         p = np.double(batch_idx + start_steps) / total_steps
#         constant = 2. / (1. + np.exp(-10 * p)) - 1
#         data3, data4 = data3.cuda(), data4.cuda()
#         label3, label4 = label3.cuda(), label4.cuda()

#         opt34.zero_grad()
        
#         # Recompute feat1 within the loop
#         with torch.no_grad():
#             feat1 = G12(data3, 0)
#             feat2 = G23(data3, feat1)
#         feat3 = G34(data3, feat1)
#         output3 = P34(feat3)
#         feat4 = G34(data4, feat1)
#         output4 = P34(feat4)
#         loss3 = loss_predict(output3, label3)
#         loss4 = loss_predict(output4, label4)
#         transfer_loss = D34(feat3, feat4, constant)
#         loss34 = loss3 + loss4 +  0.1 * transfer_loss
#         loss34.backward()
#         opt34.step()
        
#         batch_j += 1
#         if batch_j >= len(list4):
#             batch_j = 0
        
#     if epoch % interval == 0:
#         print('Train Epoch: {}\t Leaver: {}\t  loss: {:.6f}\t  loss_D: {:.6f}\t '.format(
#             epoch, '3 and 4',  loss34, transfer_loss))
# if savemodel:
#     ensure_path(model_path34)
#     torch.save(G34.state_dict(), model_path34+'/G34.pth')
#     torch.save(P34.state_dict(), model_path34+'/P34.pth')
# if loadmodel:
#     G12.load_state_dict(torch.load(model_path12+'/G12.pth'))
#     G34.load_state_dict(torch.load(model_path23+'/G34.pth'))
#     G23.load_state_dict(torch.load(model_path23+'/G23.pth'))
# for epoch in range(n_epochs45):        
#     start_steps = epoch * len(dataset4)
#     total_steps = n_epochs45 * len(dataset4)
#     batch_j = 0
#     # dataset2 and dataset3
#     for batch_idx, (data4, label4) in enumerate(dataset4):
        
#         _, (data5, label5) = list5[batch_j]
#         p = np.double(batch_idx + start_steps) / total_steps
#         constant = 2. / (1. + np.exp(-10 * p)) - 1
#         data4, data5 = data4.cuda(), data5.cuda()
#         label4, label5 = label4.cuda(), label5.cuda()

#         opt45.zero_grad()
        
#         # Recompute feat1 within the loop
#         with torch.no_grad():
#             feat1 = G12(data5, 0)
#             feat34 = G34(data5, feat1)
#             feat23 = G23(data5, feat34)
#         feat4 = G45(data4, feat23)
#         output4 = P45(feat4)
#         feat5 = G45(data5, feat23)
#         output5 = P45(feat5)
#         loss4 = loss_predict(output4, label4)
#         loss5 = loss_predict(output5, label5)
#         transfer_loss = D45(feat4, feat5, constant)
#         loss45 = loss4 + loss5 +  0.007 * transfer_loss
#         loss45.backward()
#         opt45.step()
        
#         batch_j += 1
#         if batch_j >= len(list5):
#             batch_j = 0
        
#     if epoch % interval == 0:
#         print('Train Epoch: {}\t Leaver: {}\t  loss: {:.6f}\t  loss_D: {:.6f}\t '.format(
#             epoch, '4 and 5',  loss45, transfer_loss))

# if savemodel:
#     ensure_path(model_path45)
#     torch.save(G45.state_dict(), model_path45+'/G45.pth')
#     torch.save(P45.state_dict(), model_path45+'/P45.pth')
if loadmodel:
    G12.load_state_dict(torch.load(model_path12+'/G12.pth'))
    G34.load_state_dict(torch.load(model_path23+'/G34.pth'))
    G23.load_state_dict(torch.load(model_path23+'/G23.pth'))
    G45.load_state_dict(torch.load(model_path45+'/G45.pth'))
for epoch in range(n_epochs56):
    start_steps = epoch * len(dataset6)
    total_steps = n_epochs56 * len(dataset6)
    batch_j = 0
    # dataset2 and dataset3
    for batch_idx, (data6, label6) in enumerate(dataset6):
        
        _, (data5, label5) = list5[batch_j]
        p = np.double(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1
        data6, data5 = data6.cuda(), data5.cuda()
        label6, label5 = label6.cuda(), label5.cuda()

        opt56.zero_grad()
        
        # Recompute feat1 within the loop
        with torch.no_grad():
            feat1 = G12(data6, 0)
            feat34 = G34(data6, feat1)
            feat23 = G23(data6, feat34)
            feat45 = G45(data6, feat23)
        feat5 = G56(data5, feat45)
        output5 = P56(feat5)
        feat6 = G56(data6, feat45)
        output6 = P56(feat5)
        loss6 = loss_predict(output6, label6)
        loss5 = loss_predict(output5, label5)
        transfer_loss = D56(feat6, feat5, constant)
        loss56 = loss6 + loss5 +  0.01 * transfer_loss
        loss56.backward()
        opt56.step()
        
        batch_j += 1
        if batch_j >= len(list5):
            batch_j = 0
        
    if epoch % interval == 0:
        print('Train Epoch: {}\t Leaver: {}\t  loss: {:.6f}\t  loss_D: {:.6f}\t '.format(
            epoch, '5 and 6',  loss56, transfer_loss))
# if savemodel:
#     ensure_path(model_path56)
#     torch.save(G56.state_dict(), model_path56+'/G56.pth')
#     torch.save(P56.state_dict(), model_path56+'/P56.pth')
            
# test
    
data_t = test
label = label_test
data_t = data_t.cuda()
feat12 = G12(data_t,0)
feat34 = G34(data_t, feat12)
feat23 = G23(data_t, feat34)
feat45 = G45(data_t, feat23)
feat56 = G56(data_t, feat45)
output = P56(feat56)

pred = output.cpu().detach().numpy()
label = label.detach().numpy()

pred = pred*train_std[-1] + train_mean[-1]
label = label*train_std[-1] + train_mean[-1]

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
plt.savefig("/root/dh/2024/OOD/pre.png")




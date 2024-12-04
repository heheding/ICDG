import sys
import torch
import numpy as np
import torch.utils.data as Data

def kl_divergence(p, q):
    # Ensure both arrays have the same shape
    assert p.shape == q.shape, "Arrays must have the same shape"
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    
    # Calculate KL divergence
    kl_div = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    
    return kl_div

def multivariate_train(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    new_arrays1,new_arrays2,new_arrays3,new_arrays4,new_arrays5,new_arrays6 = [],[],[],[],[],[]
    labels1,labels2,labels3,labels4,labels5,labels6 = [],[],[],[],[],[]
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)-target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    
    new_arrays1.append(data[0])
    labels1.append(labels[0])
    for i in range(1,len(data)):
        kl = kl_divergence(data[0].T,data[i].T)
        if 0 < abs(kl) <= 8:
            new_arrays1.append(data[i])
            labels1.append(labels[i])
        elif 8 < abs(kl) <= 16:
            new_arrays2.append(data[i])
            labels2.append(labels[i])
        elif 16 < abs(kl) <= 24:
            new_arrays3.append(data[i])
            labels3.append(labels[i])
        elif 24 < abs(kl) <= 32:
            new_arrays4.append(data[i])
            labels4.append(labels[i])
        elif 32 < abs(kl) <= 40:
            new_arrays5.append(data[i])
            labels5.append(labels[i])
        elif abs(kl) > 40:
            new_arrays6.append(data[i])
            labels6.append(labels[i])
    
    
    return torch.tensor(np.array(new_arrays1)), torch.tensor(np.array(labels1)),torch.tensor(np.array(new_arrays2)), torch.tensor(np.array(labels2)),torch.tensor(np.array(new_arrays3)), torch.tensor(np.array(labels3)),torch.tensor(np.array(new_arrays4)), torch.tensor(np.array(labels4)),torch.tensor(np.array(new_arrays5)), torch.tensor(np.array(labels5)),torch.tensor(np.array(new_arrays6)), torch.tensor(np.array(labels6))

def multivariate_target(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)-target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return torch.tensor(np.array(data)), torch.tensor(np.array(labels))
    
def return_dataset(df, split, week_data=False):
    features = df.drop(['time'], axis=1)
    # features = df
    features = features.rolling(window=5).mean()
    features.drop(features.head(5).index, inplace=True)

    TRAIN_SPLIT = split
    domain_piece = int(len(df) / 28)
    if week_data:
        dataset = features.values[:10000, :]
    else:
        dataset = features.values
    train_split = int(dataset.shape[0] * TRAIN_SPLIT)
    train_min = dataset[:].min(0)
    train_max = dataset[:].max(0)
    train_cha = train_max - train_min
    dataset = (dataset-train_min)/(train_cha)

    past_history = 30
    future_target = 1
    STEP = 1

    x_train1, y_train1,x_train2, y_train2,x_train3, y_train3,x_train4, y_train4,x_train5, y_train5,x_train6, y_train6 = multivariate_train(
        dataset[:, :-1], dataset[:, -1], 0, train_split, past_history, future_target, STEP, single_step=True)
    x_val, y_val = multivariate_target(
        dataset[:, :-1], dataset[:, -1], train_split, None, past_history, future_target, STEP, single_step=True)

    return train_min, train_cha, x_train1, y_train1,x_train2, y_train2,x_train3, y_train3,x_train4, y_train4,x_train5, y_train5,x_train6, y_train6, x_val, y_val


def dataset_read(source, split_s, batch_size):

    train_mean, train_std, train1, label_train1,train2, label_train2,train3, label_train3,train4, label_train4,train5, label_train5,train6, label_train6, test, label_test = return_dataset(source, split_s, week_data=True)


    dataset_train1 = Data.TensorDataset(train1, label_train1)

    dataset1 = torch.utils.data.DataLoader(
        dataset_train1,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    dataset_train2 = Data.TensorDataset(train2, label_train2)

    dataset2 = torch.utils.data.DataLoader(
        dataset_train2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    dataset_train3 = Data.TensorDataset(train3, label_train3)

    dataset3 = torch.utils.data.DataLoader(
        dataset_train3,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    dataset_train4 = Data.TensorDataset(train4, label_train4)

    dataset4 = torch.utils.data.DataLoader(
        dataset_train4,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    dataset_train5 = Data.TensorDataset(train5, label_train5)

    dataset5 = torch.utils.data.DataLoader(
        dataset_train5,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    dataset_train6 = Data.TensorDataset(train6, label_train6)

    dataset6 = torch.utils.data.DataLoader(
        dataset_train6,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    


    return train_mean, train_std, dataset1,dataset2,dataset3,dataset4,dataset5,dataset6, test, label_test

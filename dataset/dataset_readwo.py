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
    
    
    return torch.tensor(np.array(data)), torch.tensor(np.array(labels))

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
    train_split = int(9000)
    train_min = dataset[:].min(0)
    train_max = dataset[:].max(0)
    train_cha = train_max - train_min
    dataset = (dataset-train_min)/(train_cha)

    past_history = 30
    future_target = 1
    STEP = 1

    x_train, y_train = multivariate_train(
        dataset[:, :-1], dataset[:, -1], 0, train_split, past_history, future_target, STEP, single_step=True)
    x_val, y_val = multivariate_target(
        dataset[:, :-1], dataset[:, -1], train_split, 9950, past_history, future_target, STEP, single_step=True)

    return train_min, train_cha, x_train, y_train, x_val, y_val


def dataset_read(source, split_s, batch_size):

    train_s_mean, train_s_std, train_source, s_label_train, test_source, s_label_test = return_dataset(source, split_s, week_data=True)


    dataset_source = Data.TensorDataset(train_source, s_label_train)

    dataset_s = torch.utils.data.DataLoader(
        dataset_source,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    


    return train_s_mean, train_s_std, dataset_s, test_source, s_label_test

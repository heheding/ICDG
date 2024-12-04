import sys
import torch
import numpy as np
import torch.utils.data as Data
from sklearn.decomposition import PCA


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
    # features = df
    features = df.rolling(window=5).mean()
    features.drop(features.head(5).index, inplace=True)

    TRAIN_SPLIT = split
    domain_piece = int(len(df) / 28)
    if week_data:
        dataset = features.values[:, :]
    else:
        dataset = features.values
    train_split = int(12000)
    train_min = dataset[:].min(0)
    train_max = dataset[:].max(0)
    train_cha = train_max - train_min
    dataset = (dataset-train_min)/(train_cha)

    past_history = 30
    future_target = 1
    STEP = 1
    
    # Convert x_train from torch tensor to numpy array for PCA
    x_train_np = dataset[:, :-2]

    # Fit PCA
    n_components = 5  # You can set the number of components you want
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_np)

    # Concatenate the PCA transformed features with the original x_train
    data2 = np.concatenate((x_train_np, x_train_pca), axis=1)

    x_train, y_train = multivariate_target(
        data2, dataset[:, -2], 0, train_split, past_history, future_target, STEP, single_step=True)
    
    
    x_val, y_val = multivariate_target(
        data2, dataset[:, -2], train_split, 12990, past_history, future_target, STEP, single_step=True)

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

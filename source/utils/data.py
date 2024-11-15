import random
from sklearn.model_selection import KFold
import torch
from torch_geometric.transforms import BaseTransform

class DataToFloat(BaseTransform):
    def __call__(self, data):
        data.x = data.x.to(torch.float32)
        return data
    
def get_train_val_test_datasets(data, seed, n_folds, fold_id, ratio=0.85):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_splits = [k for k in kf.split(data)]
    trainval_indices, test_indices = all_splits[fold_id]
    trainval_indices, test_indices = trainval_indices.tolist(), test_indices.tolist()
    trainval_dataset = [data[i] for i in trainval_indices]
    test_dataset = [data[i] for i in test_indices]

    indices = list(range(len(trainval_dataset)))
    random.seed(seed)
    random.shuffle(indices)
    split_index = int(ratio * len(indices))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    train_dataset = [trainval_dataset[i] for i in train_indices]
    val_dataset = [trainval_dataset[i] for i in val_indices]

    return train_dataset, val_dataset, test_dataset
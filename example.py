import random
import os

# import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader

# from src.core.mongo import sync_database
# from app.core.settings import billion_settings
from app.helpers import (  # load_dataset_from_file,; load_dataset_with_features,
    get_triplets,
    get_patterns,
)
from app.net import BillionModel

# import traceback
# from typing import Union


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def start():
    """
    Method for app starting

    Returns:
        None
    """
    optuna_params = {}

    # print(optuna_params)
    billion = BillionModel(
        margin=optuna_params.get('margin', 1.0),
        epochs=optuna_params.get('epochs', 12),
        embedding_dim=optuna_params.get('embedding_dim', optuna_params.get('pattern_size', 90) * 10),
        pattern_size=optuna_params.get('pattern_size', 90),
        learning_rate=optuna_params.get('learning_rate', 0.00009470240447408595),
    )
    # # generate dataset
    # train_data = self.dataset[:6000]
    # val_data = self.dataset[6000:9000]
    #
    # train_data_features = self.dataset_features[:6000]
    # val_data_features = self.dataset_features[6000:9000]

    # print(train_data)
    # print(train_data_features)
    #
    # print(val_data)
    # print(val_data_features)

    # form dataset (format triplets)
    # here we can specify batch size to speed up training
    data_list = billion.scaler.fit_transform(train_data_features.values.tolist())  #
    close_price = train_data["Close"].values.tolist()

    # generate patterns
    patterns, no_patterns, pattern_indexes, no_pattern_indexes = get_patterns(
        close_price=close_price,
        fully_data=data_list,
        extr_window=optuna_params.get('extr_window', 69),
        until_profit=optuna_params.get('until_profit', 10),
        profit_value=optuna_params.get('profit_value', 1.8),
        pattern_size=optuna_params.get('pattern_size', 90),
    )
    # print(len(no_patterns), len(patterns))
    os.environ['PYTHONHASHSEED'] = str(1300)
    random.seed(1300)

    if len(no_patterns) == 0 or len(patterns) == 0:
        return {'loss': 0, 'accuracy': 0, 'precision': 0, 'f-score': 0}

    # balance data
    if len(no_patterns) > len(patterns):
        diff = len(no_patterns) - len(patterns)
        patterns = patterns + random.choices(patterns, k=diff)

    if len(patterns) > len(no_patterns):
        diff = len(patterns) - len(no_patterns)
        no_patterns = no_patterns + random.choices(no_patterns, k=diff)

    # print(np.array(patterns), np.array(no_patterns))

    # generate triplets
    # print('patterns', patterns, no_patterns)
    triplets = get_triplets(patterns, no_patterns)
    # print('train triplets', triplets)
    # return 0

    triplets = torch.from_numpy(triplets).type(torch.float32)

    # generate train dataset
    train_dataset = torch.utils.data.TensorDataset(triplets[0, :], triplets[1, :], triplets[2, :])
    train_dataloader = DataLoader(train_dataset, batch_size=256)

    # generate validation dataset
    data_list = billion.scaler.transform(val_data_features.values.tolist())  #
    close_price = val_data["Close"].values.tolist()

    # get all patterns for the validation period
    patterns_val, no_patterns_val, _, _ = get_patterns(
        close_price=close_price,
        fully_data=data_list,
        extr_window=optuna_params.get('extr_window', 69),
        until_profit=optuna_params.get('until_profit', 10),
        profit_value=optuna_params.get('profit_value', 1.8),
        pattern_size=optuna_params.get('pattern_size', 90),
    )

    patterns_val = np.asarray(patterns_val)
    no_patterns_val = np.asarray(no_patterns_val)

    all_patterns = np.concatenate((patterns_val, no_patterns_val))

    # now let's generate targets
    # for pattern sampling it is of course always signal 1
    patterns_targets = np.array([1] * patterns_val.shape[0])
    # for non-patterns always signal 0
    no_patterns_targets = np.array([0] * no_patterns_val.shape[0])

    targets = np.concatenate((patterns_targets, no_patterns_targets))

    # target and pattern shapes must be the same
    assert targets.shape[0] == all_patterns.shape[0]

    # fix seed before shuffling
    np.random.seed(1300)
    # generate common index
    indices = np.arange(targets.shape[0])
    np.random.shuffle(indices)

    # we don't join all the lists, so we use
    # a common index to mix them all at once.
    targets = targets[indices]
    all_patterns = all_patterns[indices]

    all_patterns = torch.from_numpy(all_patterns).type(torch.float32)
    targets = torch.from_numpy(targets).type(torch.float32)

    # form dataset (format input data - correct answer)
    val_dataset = torch.utils.data.TensorDataset(all_patterns, targets)
    val_dataloader = DataLoader(val_dataset)

    patterns = np.asarray(patterns)
    no_patterns = np.asarray(no_patterns)

    # print(optuna_params)
    metrics = billion.train(
        train_dataloader,
        (
            torch.from_numpy(patterns[0][None, ...]).type(torch.float32),
            torch.from_numpy(no_patterns[0][None, ...]).type(torch.float32),
        ),
        (val_dataloader, targets.to(device)),
    )
    # print(hyperparams)


if __name__ == '__main__':
    start()

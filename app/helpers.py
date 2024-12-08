import random
import os

import pandas as pd
import torch
import numpy as np
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extremes_window(arr: list, window: int, until_profit: int) -> tuple:
    """
    Generates max and min pointers
    using sliding window

    Returns:
    Two arrays max and min
    """
    indexes_max = []
    indexes_min = []

    for i in range(len(arr) - window - until_profit):
        window_arr = arr[i : i + window]

        if max(window_arr) == window_arr[-1]:
            indexes_max.append(i + window - 1)

        if min(window_arr) == window_arr[-1]:
            indexes_min.append(i + window - 1)

    return indexes_max, indexes_min


def get_patterns(
    close_price: np.array,
    fully_data: pd.DataFrame,
    extr_window: int,
    until_profit: int,
    profit_value: float,
    pattern_size: int,
) -> tuple:
    """
    Generates patterns and no-patterns. We iterate on the indices of maxs
    and mins and get a backward window (of size pattern_size),
    provided that after N (until_profit) bars from the extremum we get a profit (profit_value)

    Args:
        close_price: Array for pattern searching
        fully_data: Array for pattern generating
        extr_window: Inside this window will find extremes
        until_profit: Inside this window will find extremes
        profit_value: Inside this window will find extremes
        pattern_size: Inside this window will find extremes

    Returns:
        Tuple P, NP, their indices
    """
    max_indexes, min_indexes = extremes_window(close_price, extr_window, until_profit)

    # print('max_indexes', max_indexes)
    # print('min_indexes', min_indexes)
    # Iterate on the indices of maxs and mins. If after N (until_profit) bars the profit
    # has reached the desired one - refer the index to the pattern cluster
    pattern_indexes = [
        max_index
        for max_index in max_indexes
        if close_price[max_index] - min(close_price[max_index : max_index + until_profit + 1]) > profit_value
    ] + [
        min_index
        for min_index in min_indexes
        if max(close_price[min_index : min_index + until_profit + 1]) - close_price[min_index] > profit_value
    ]

    # print(pattern_indexes)

    # Form the patterns themselves. Go through the indices
    # and get the backward window (of size pattern_size)
    patterns = [
        [fully_data[pattern_index - pattern_size + 1 : pattern_index + 1]]
        for pattern_index in pattern_indexes
        if len(close_price[pattern_index - pattern_size + 1 : pattern_index + 1]) == pattern_size
    ]

    # Iterate on the indices of maxs and mins. If after N (until_profit) bars the profit
    # hasn't reached the desired one - refer the index to the NO-pattern cluster
    no_pattern_indexes = [max_index for max_index in max_indexes if max_index not in pattern_indexes] + [
        min_index for min_index in min_indexes if min_index not in pattern_indexes
    ]

    # Form the patterns themselves. Go through the indices
    # and get the backward window (of size pattern_size)
    no_patterns_val = [
        [fully_data[no_pattern_index - pattern_size + 1 : no_pattern_index + 1]]
        for no_pattern_index in no_pattern_indexes
        if len(close_price[no_pattern_index - pattern_size + 1 : no_pattern_index + 1]) == pattern_size
    ]
    return patterns, no_patterns_val, pattern_indexes, no_pattern_indexes


def get_apn(a: list, b: list) -> tuple:
    """
    Generates anchors, positivies, neagtives
    using target and object arrays

    Returns:
        Tuple A, P, N
    """
    # use first element as anchor and generate array
    anchors = [np.mean(a, axis=0)] * len(a)
    positives = a.copy()  ## np.random.choice(a, size=1, replace=False)  #
    negatives = b.copy()  # np.random.choice(b, size=len(a))  #, replace=False)

    # fix seed before shuffling
    os.environ['PYTHONHASHSEED'] = str(1300)
    random.seed(1300)

    # shuffle both arrays to balance series
    random.shuffle(positives)
    random.shuffle(negatives)

    anchors = np.asarray(anchors)
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)

    return anchors, positives, negatives


def get_triplets(patterns: list, no_patterns: list) -> np.array:
    """
    Generates triplets (https://en.wikipedia.org/wiki/Triplet_loss)
    for training the Siamese network

    Returns:
    Tuple A, P, N
    """
    p_anchors, p_positives, p_negatives = get_apn(patterns, no_patterns)

    np_anchors, np_positives, np_negatives = get_apn(no_patterns, patterns)

    anchors = np.concatenate((p_anchors, np_anchors))
    positives = np.concatenate((p_positives, np_positives))
    negatives = np.concatenate((p_negatives, np_negatives))

    # sample shape should be the same
    assert anchors.shape[0] == positives.shape[0] == negatives.shape[0]

    # fix seed before shuffling
    np.random.seed(1300)

    # generate common index
    indices = np.arange(anchors.shape[0])
    np.random.shuffle(indices)

    # we don't join all the lists, so we use
    # a common index to mix them all at once.
    anchors = anchors[indices]
    positives = positives[indices]
    negatives = negatives[indices]

    return np.stack((anchors, positives, negatives))


def distance_function(x: np.array, y: np.array) -> float:
    return 1.0 - F.cosine_similarity(x, y)


def get_validation_outputs(
    net, anchor_pattern: torch.Tensor, anchor_no_pattern: torch.Tensor, dataloader: torch.utils.data.TensorDataset
) -> torch.Tensor:
    """
    Goes through the val dataset and generates a list of preds
    to compare with the targets

    Args:
        net: Torch model
        anchor_pattern: Anchor for pattern
        anchor_no_pattern: Anchor for no pattern
        dataloader: Torch val dataset

    Returns:
        Tensor of preds ([0, 1 ... 1, 1])
    """
    # load our anchors to get embeddings to get an accessory distances
    anchor_pattern = net.forward_once(anchor_pattern.to(device))
    anchor_no_pattern = net.forward_once(anchor_no_pattern.to(device))
    outputs = []

    for i, (input_data, target) in enumerate(dataloader, 0):
        # get embedding for input data
        output = net.forward_once(input_data.to(device))

        # calculate distances between inputs and anchors
        pattern_dist = distance_function(anchor_pattern, output).item()
        no_pattern_dist = distance_function(anchor_no_pattern, output).item()

        # an object is a pattern if its embedding is close
        # to the embedding of the pattern anchor and farther
        # away from the non-pattern anchor
        if pattern_dist > no_pattern_dist:
            outputs.append(0)
        else:
            outputs.append(1)

    return torch.Tensor(outputs).to(device)


def load_dataset_from_file(path: str, index_format: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Datetime")
    df.index = pd.to_datetime(df.index, format=index_format)
    df = df.fillna(method='ffill')
    df = df.dropna()
    return df


def featurized(df) -> pd.DataFrame:
    # hyperparams = toml.load("hyperparams.toml")

    # df = load_dataset_from_file("input/15minData.csv", index_format=None)
    # df = df.fillna(method='ffill')
    # df = df.dropna()
    # print(df)

    # generate features
    # features = df.rename(columns={'Volume': 'GC_Volume', 'Close': 'GC_Close'})
    # features = df.rename(
    #     columns={
    #         "xau_Open": "xau_Open",
    #         "xau_High": "xau_High",
    #         "xau_Low": "xau_Low",
    #         "xau_close": "xau_close",
    #         "xau_Volume": "xau_Volume"
    #     }
    # )
    features = df

    features['xau_co'] = features['xau_close'] - features['xau_open']
    features['xau_vd'] = features['xau_volume'].diff()

    features['gvz_co'] = features['gvz_close'] - features['gvz_open']
    features['tnx_co'] = features['tnx_close'] - features['tnx_open']
    features['dxy_co'] = features['dxy_close'] - features['dxy_open']

    # features['GC_HL'] = features['High'] - features['Low']
    # features['GVZ_HL'] = features['GVZ_High'] - features['GVZ_Low']
    # features['TNX_HL'] = features['TNX_High'] - features['TNX_Low']
    # features['DXY_HL'] = features['DXY_High'] - features['DXY_Low']

    features = features[
        ['xau_close', 'xau_co', 'xau_volume', 'xau_vd', 'gvz_close', 'gvz_co', 'tnx_close', 'tnx_co', 'dxy_close', 'dxy_co']
    ]
    features = features.fillna(0)
    return features

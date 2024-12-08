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
    positives = a.copy()
    negatives = b.copy()

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

    assert p_anchors.shape[0] == p_positives.shape[0] == p_negatives.shape[0]
    assert np_anchors.shape[0] == np_positives.shape[0] == np_negatives.shape[0]

    anchors = np.concatenate((p_anchors, np_anchors))
    positives = np.concatenate((p_positives, np_positives))
    negatives = np.concatenate((p_negatives, np_negatives))

    targets = np.array([1] * p_anchors.shape[0] + [0] * np_anchors.shape[0])
    targets = targets.reshape((targets.shape[0], 1))

    # sample shape should be the same
    assert anchors.shape[0] == positives.shape[0] == negatives.shape[0] == targets.shape[0]

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
    targets = targets[indices]

    return np.stack((anchors, positives, negatives)), targets


def distance_function(x: np.array, y: np.array) -> float:
    """
    Calculates cos distance between vectors
    Args:
        x: Vector One
        y: Vector Two

    Returns:
        Cos distance
    """
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
    """
    Loads dataset using datetime as an index
    Args:
        path: Path to dataframe
        index_format: Datetime index format

    Returns:
        Pandas Dataframe
    """
    df = pd.read_csv(path, index_col="Datetime")
    df.index = pd.to_datetime(df.index, format=index_format)
    df = df.fillna(method='ffill')
    df = df.dropna()
    return df


def featurized(df) -> pd.DataFrame:
    """
    Converts source dataframe into features
    Args:
        df: Source dataframe (Open, High, Low, Close (XAU, DXY, TNX, GVZ))

    Returns:
        Features for prediction
    """
    df = df.rename(
        columns={'xau_open': 'open', 'xau_close': 'close', 'xau_high': 'high', 'xau_low': 'low', 'xau_volume': 'volume'}
    )

    df = df.rename(
        columns={
            'Volume': 'gc_volume',
            'Close': 'gc_close',
            'Open': "gc_open",
            'GVZ_Open': "gvz_open",
            'GVZ_High': "gvz_high",
            'GVZ_Low': "gvz_low",
            'GVZ_Close': "gvz_close",
            'TNX_Open': "tnx_open",
            'TNX_High': "tnx_high",
            'TNX_Low': "tnx_low",
            'TNX_Close': "tnx_close",
            'DXY_Open': "dxy_open",
            'DXY_High': "dxy_high",
            'DXY_Low': "dxy_low",
            'DXY_Close': "dxy_close",
        }
    )

    df['GC_CO'] = df['gc_close'] - df['gc_open']
    df['GC_VD'] = df['gc_volume'].diff()

    df['GVZ_CO'] = df['gvz_close'] - df['gvz_open']
    df['TNX_CO'] = df['tnx_close'] - df['tnx_open']
    df['DXY_CO'] = df['dxy_close'] - df['dxy_open']

    df['GVZ_Close'] = df['gvz_close'].diff()
    df['TNX_Close'] = df['tnx_close'].diff()
    df['DXY_Close'] = df['dxy_close'].diff()

    df = df[
        ['gc_close', 'gc_volume', 'GC_VD', 'GC_CO', 'GVZ_Close', 'GVZ_CO', 'TNX_Close', 'TNX_CO', 'DXY_Close', 'DXY_CO']
    ]
    df = df.fillna(0)
    return df

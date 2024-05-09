import logging
import pathlib
from typing import List, NoReturn, Optional, Tuple

import numpy
import pandas
import scipy.io
import torch
from torch.utils import data as tor_data


class GramSlicer:
    """Turn a gram into a :class:`numpy.ndarray` of gram snippets.

    :param length: Snippet length.
    :param stride: Slicing stride.
    """
    
    def __init__(self, length: int, stride: int):
        self.length = length
        self.stride = stride
    
    def __call__(self, gram: numpy.ndarray) -> numpy.ndarray:
        snippets = []
        for i in range(0, gram.shape[2] - (self.length - self.stride), self.stride):
            snippet = gram[:, :, i:i + self.length]
            # padding at the end
            if snippet.shape[2] != self.length:
                snippet = numpy.concatenate(
                        (snippet, numpy.zeros((*(snippet.shape[0:2]), self.length - snippet.shape[2]), snippet.dtype)),
                        axis = 2)
            
            snippets.append(snippet)
        return numpy.array(snippets, copy = True)


class MyBaseDataset(tor_data.Dataset):
    """My base dataset.
    
    The content got from the dataset is `(sample, room label)`.
    
    :param data_dir: The path to the folder that contains all data.
    :param data: The DataFrame that contains entries which should be in this dataset.
    :param class_num: The number of classes.
    :param gram_type: which type of gram is used for features.
    :param slicer: The slicer that turn a gram into a :class:`numpy.ndarray` of gram snippets. If not provided,
        grams won't be sliced.
    :param additional_info: If true, the dataset will also provide user and position labels.
    """
    
    def __init__(self,
                 data_dir: pathlib.Path, data: pandas.DataFrame,
                 class_num: int,
                 gram_type: str, slicer: GramSlicer,
                 additional_info: bool):
        self.data_dir = data_dir
        
        self.sample_names: numpy.ndarray = data['gram'].to_numpy()
        self.room_labels: numpy.ndarray = data['room'].to_numpy()
        
        self.gram_type = gram_type
        self.gram_name = 'gram_' + self.gram_type
        self.slicer = slicer
        
        self.class_num = class_num
        self.additional_info = additional_info
    
    def __len__(self) -> int:
        return len(self.room_labels)
    
    def __getitem__(self, ind) -> Tuple[torch.Tensor, torch.Tensor]:
        # read gram mat
        sample_name = self.sample_names[ind]
        sample_path = self.data_dir / sample_name
        sample = scipy.io.loadmat(sample_path, appendmat = False, variable_names = [self.gram_name])[self.gram_name]
        if self.slicer is not None:
            sample = self.slicer(sample)
        sample = torch.from_numpy(sample)
        
        room_label = self.room_labels[ind]
        
        return sample, room_label


class MyDataset(MyBaseDataset):
    """My dataset.
    
    The content got from the dataset is `(sample, activity label, one-hot activity label, room label,
    optional[user label, position label])`.
    
    :param data_dir: The path to the folder that contains all data.
    :param data: The DataFrame that contains entries which should be in this dataset.
    :param class_num: The number of classes.
    :param gram_type: which type of gram is used for features.
    :param slicer: The slicer that turn a gram into a :class:`numpy.ndarray` of gram snippets. If not provided,
        grams won't be sliced.
    :param additional_info: If true, the dataset will also provide user and position labels.
    """
    
    def __init__(self,
                 data_dir: pathlib.Path, data: pandas.DataFrame,
                 class_num: int,
                 gram_type: str, slicer: Optional[GramSlicer] = None,
                 additional_info: bool = False):
        super(MyDataset, self).__init__(data_dir, data, class_num, gram_type, slicer, additional_info)
        
        self.act_labels: numpy.ndarray = data['activity'].to_numpy()
        if self.additional_info:
            self.user_labels: numpy.ndarray = data['user'].to_numpy()
            self.pos_labels: numpy.ndarray = data['position'].to_numpy()
    
    def __getitem__(self, ind) -> Tuple[torch.Tensor, ...]:
        if torch.is_tensor(ind):
            ind = ind.tolist()
        
        sample, room_label = super(MyDataset, self).__getitem__(ind)
        
        act_label = self.act_labels[ind]
        act_onehot_label = numpy.array([1 if i == act_label else 0 for i in range(self.class_num)]).astype(numpy.int64)
        act_onehot_label = torch.from_numpy(act_onehot_label).type(sample.dtype)
        
        returns = [sample, act_label, act_onehot_label, room_label]
        
        if self.additional_info:
            user_label = self.user_labels[ind]
            # user_label = torch.from_numpy(user_label)
            pos_label = self.pos_labels[ind]
            # pos_label = torch.from_numpy(pos_label)
            returns.extend([user_label, pos_label])
        
        return tuple(returns)


class MyRandomActivityDataset(MyBaseDataset):
    r"""My random activity dataset.
    
    The content got from the dataset is `(sample, activity label, one-hot activity label, room label,
    optional[user label])`.
    Activity label will be 9, onehot activity label will be all zero.
    
    :param data_dir: The path to the folder that contains all data.
    :param data: The DataFrame that contains entries which should be in this dataset.
    :param class_num: The number of classes.
    :param gram_type: which type of gram is used for features.
    :param slicer: The slicer that turn a gram into a :class:`numpy.ndarray` of gram snippets. If not provided,
        grams won't be sliced.
    :param additional_info: If true, the dataset will also provide user and position labels.
    """
    
    def __init__(self,
                 data_dir: pathlib.Path, data: pandas.DataFrame,
                 class_num: int,
                 gram_type: str, slicer: Optional[GramSlicer] = None,
                 additional_info: bool = False):
        super(MyRandomActivityDataset, self).__init__(data_dir, data, class_num, gram_type, slicer, additional_info)
        
        if self.additional_info:
            self.user_labels: numpy.ndarray = data['user'].to_numpy()
    
    def __len__(self) -> int:
        return len(self.room_labels)
    
    def __getitem__(self, ind) -> Tuple[torch.Tensor, ...]:
        if torch.is_tensor(ind):
            ind = ind.tolist()
        
        sample, room_label = super(MyRandomActivityDataset, self).__getitem__(ind)
        
        act_label = numpy.int64(9)
        act_onehot_label = numpy.array([0 for _ in range(self.class_num)]).astype(numpy.int64)
        act_onehot_label = torch.from_numpy(act_onehot_label).type(sample.dtype)
        
        returns = [sample, act_label, act_onehot_label, room_label]
        
        if self.additional_info:
            user_label = self.user_labels[ind]
            returns.extend([user_label])
        
        return tuple(returns)


def get_datasets(
        data_dir: pathlib.Path,
        class_num: int,
        gram_type: str,
        rooms_list: List[List[int]],
        key: Optional[str] = 'data',
        additional_labels: Optional[bool] = False,
        slicing: Optional[bool] = False,
        slice_length: Optional[int] = None,
        slice_stride: Optional[int] = None) -> Tuple[tor_data.Dataset, ...]:
    """
    Get datasets from the data folder.
    
    todo multiple domain category.
    
    :param data_dir: The path to the folder that contains all data.
    :param key: The pandas object group identifier of data list that will get loaded. Must be `'data'` or `'random'`.
    :param additional_labels: Should the dataset return additional labels?
    :param class_num: The number of classes.
    :param gram_type: which type of gram is used for features.
    :param rooms_list: A list of list of rooms that each returned dataset will contain.
    :param slicing: Slice the gram or not.
    :param slice_length: Snippet length.
    :param slice_stride: Slicing stride.
    :return: `len(rooms_list)` datasets.
    """
    logger = logging.getLogger('data_loader')
    
    if slicing:
        if slice_length is None or slice_stride is None:
            message = "Gram can not be sliced without slice length and slice stride."
            logger.error(message)
            raise ValueError(message)
        logger.debug("Gram snippets length is {}, stride is {}.".format(slice_length, slice_stride))
        slicer = GramSlicer(slice_length, slice_stride)
    else:
        logger.debug("Gram won't be sliced.")
        slicer = None
    
    if key == 'data':
        logger.info("Load activity data.")
        dataset_class = MyDataset
    elif key == 'random':
        logger.info("Load random activity data.")
        dataset_class = MyRandomActivityDataset
    else:
        message = "Unknown pandas object group identifier {}.".format(key)
        logger.error(message)
        raise ValueError(message)
    
    # read data list
    data: pandas.DataFrame = pandas.read_hdf(data_dir / 'data_list.hf', key = key)
    if not isinstance(data, pandas.DataFrame):
        message = "The data stored in data_list.hf is a {} instead of pandas.DataFrame".format(type(data))
        logger.error(message)
        raise TypeError(message)
    
    logger.debug("Use {} gram as input features.".format(gram_type))
    
    # split data list based on rooms_list
    num = len(rooms_list)
    datasets = []
    logger.debug("Generate {} datasets.".format(num))
    for i in range(num):
        logger.debug("Dataset {} contains data from rooms {}.".format(i, rooms_list[i]))
        data_1 = data.loc[data['room'].isin(rooms_list[i]), :]
        dataset_1 = dataset_class(data_dir, data_1, class_num, gram_type, slicer, additional_labels)
        datasets.append(dataset_1)
    
    return tuple(datasets)


def split_dataset(dataset: tor_data.Dataset, split_ratio: float = 0.1) -> Tuple[tor_data.Dataset, tor_data.Dataset]:
    """
    Randomly split the dataset into two datasets.
    
    :param dataset: The dataset to be split.
    :param split_ratio: What percentage of data is kept for the first dataset.
    :return: Two datasets.
    """
    
    # calculate the size of each subset
    dataset_size = len(dataset)
    dataset1_size = int(numpy.floor(split_ratio * dataset_size))
    dataset2_size = dataset_size - dataset1_size
    
    dataset1, dataset2 = tor_data.random_split(dataset, [dataset1_size, dataset2_size])
    
    return dataset1, dataset2


def get_data_loader(dataset: tor_data.Dataset, batch_size: int, drop_last: bool = False,
                    worker_num: int = 1) -> tor_data.DataLoader:
    """
    Return a :class:`torch.utils.data.DataLoader` with proper worker random seeding, pinned memory, and data reshuffle
    every epoch.

    :param dataset: Dataset from which to load the data.
    :param batch_size: How many samples per batch to load.
    :param drop_last: Drop the last incomplete batch, if the dataset size is not divisible by the batch size.
    :param worker_num: how many sub-processes to use for data loading. ``0`` means that the data will be loaded in
        the main process.
    :return: a DataLoader.
    """
    
    def _worker_init_fn(worker_id: int):
        """
        Set numpy seed for the DataLoader worker.
        """
        numpy.random.seed(tor_data.get_worker_info().seed % 1000000000 + worker_id)
    
    return tor_data.DataLoader(
            dataset, batch_size = batch_size,
            shuffle = True, drop_last = drop_last,
            num_workers = worker_num, worker_init_fn = _worker_init_fn, pin_memory = True)


def __test() -> NoReturn:
    """
    Test.
    """
    from configs import default_configs as conf
    logging.basicConfig(level = logging.DEBUG)
    set1, set2 = get_datasets(
            conf.data_dir,
            class_num = conf.class_num,
            gram_type = conf.gram_type,
            rooms_list = [[i for i in range(conf.data_count['room']) if i != 3], [3]],
            slicing = conf.slicing,
            slice_length = conf.slice_length, slice_stride = conf.slice_stride, additional_labels = True)
    print(len(set1))
    print(len(set2))
    loader1 = get_data_loader(set1, conf.batch_size, 0)
    batch = next(iter(loader1))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)
    set1, = get_datasets(
            conf.data_dir,
            class_num = conf.class_num,
            gram_type = conf.gram_type,
            rooms_list = [[i for i in range(conf.data_count['room'])]],
            key = 'random',
            slicing = conf.slicing,
            slice_length = conf.slice_length, slice_stride = conf.slice_stride, additional_labels = True)
    print(len(set1))
    loader1 = get_data_loader(set1, conf.batch_size, 0)
    batch = next(iter(loader1))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)


if __name__ == '__main__':
    __test()

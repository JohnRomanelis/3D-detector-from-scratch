import torch 
from torch.utils.data import Dataset
import numpy as np 
import pickle
from pathlib import Path

class DatasetTemplate(Dataset):

    def __init__(self, 
                 root_path, 
                 info_path, 
                 class_names=None,
                 transforms=[],
                 num_point_features=4):
        
        # saving the root path
        self._root_path = root_path

        # storing the classes for the dataset
        self._class_names = class_names

        # loading the info files 
        try:
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
        except:
            print("Problem with loading the info files")
            print("Make sure you have generated the info files")
            raise Exception
        
        self._infos = infos
        print("Remaining Number of Infos: ", len(self._infos))

        # handling transforms
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms    

        self.num_point_features = num_point_features    


    def __len__(self):
        return len(self._infos)


    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)

        example = input_dict.copy()
        for transform in self.transforms:
            example = transform(example)
        
        # adding metadata
        example['metadata'] = {}
        self.add_metadata_to_example(example, input_dict)

        # if anchor mask is included -> change its type to uint8
        #if "anchors_mask" in example:
        #    example['anchors_mask'] = example['anchors_mask'].astype(np.uint8)

        return example


    def add_metadata_to_example(self, example, input_dict):
        # copies metadata from example to the new dictionart
        raise NotImplementedError


    def get_sensor_data(self, idx):
        raise NotImplementedError

import hydra
import os
from pathlib import Path

# searching for the given path
# raise ValueError is path is not found
def configure_path(path, keyword):

    # searching for direct path
    if os.path.isdir(path):
        return path
    
    # searching for relative path 
    elif path=='default':
        # data should be in the /data folder and the subfolder 
        # should be named as the dataset, (aka keyword)
        data_path = Path(hydra.utils.get_original_cwd()) 
        data_path = data_path / Path("data") / Path(keyword)
        if os.path.isdir(data_path):
            return data_path
        else: 
            raise ValueError
    else:
        # not using the default path, 
        # but path describes the relative path 
        #data_path = Path(hydra.utils.get_original_cwd())
        data_path = hydra.utils.get_original_cwd() / Path(path)
        if os.path.isdir(data_path):
            return data_path
        else:
            raise ValueError 





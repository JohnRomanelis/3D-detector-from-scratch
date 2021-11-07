from pathlib import Path

def int2constString(num, string_len=6):
    '''
        Input : num (intiger)

        Output : a string with a constant length of string_len
        
            e.g  num=1, string_len=6   =>   "000001"
    '''
    
    return str(num).zfill(string_len) 

def remove_dontcare(image_anno):
    image_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x != 'DontCare'
    ]

    for key in image_anno.keys():
        image_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices]
        )

    return image_filtered_annotations

def get_path(idx, main_path, training, data_type, relative_path=None):

    """
    TODO: add implementation for relative path 
          (may be working with current implementation)
    """


    # making index a string with 6 digits as used in kitti
    # and turning that string to a path
    idx = Path(int2constString(idx))

    if not isinstance(main_path, Path):
        main_path = Path(main_path)

    if training:
        task = "training"
    else: 
        task = "testing"
    # constructing main path
    data_path = Path(main_path) / task / data_type / idx

    # addign file suffix
    if data_type=="velodyne":
        data_path = str(data_path) + ".bin"
    elif data_type=="calib":
        data_path = str(data_path) + ".txt"
    elif data_type=="image_2":
        data_path = str(data_path) + ".png"
    elif data_type=="label_2":
        data_path = str(data_path) + ".txt"
    else:
        print("""Unknown data type. Should be one of the following options: 
            - velodyne
            - calib
            - label_2
            - image_2
         """)
        raise ValueError

    return data_path

def read_split_file(path):
    # reading a txt file containing the indexes of the frames
    # that belong to different kitti tasks
    # (i.e. training, validation, trainval, testing)
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
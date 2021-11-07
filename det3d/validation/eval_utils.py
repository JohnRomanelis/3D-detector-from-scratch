from pathlib import Path

def int2constString(num, string_len=6):
    '''
        Input : num (intiger)

        Output : a string with a constant length of string_len
        
            e.g  num=1, string_len=6   =>   "000001"
    '''
    
    return str(num).zfill(string_len)



def save_detection_results_to_txt(path, pred_dicts, category='Car'):
    """
    Args: 
         - pred_dicts: list with network predictions
    """
    # TODO: use pred_dict['label_preds'] to replace category

    # getting batch size
    batch_size = len(pred_dicts)
    
    for b in range(batch_size):
        
        pred_dict = pred_dicts[b]

        idx = int2constString(pred_dict['metadata']["image_idx"])
        num_detections = pred_dict['box3d_lidar'].shape[0]

        filepath = Path(path) / (idx + '.txt')
        with open(filepath, 'w') as f:

            for i in range(num_detections):
                f.write(category + " ")
                # we don't care about these values
                for j in range(7):
                    f.write("-10 ")
                
                # getting bounding box info
                location = pred_dict['box3d_lidar'][i, :3].cpu().numpy()
                dimensions = pred_dict['box3d_lidar'][i, 3:6].cpu().numpy()
                angle = pred_dict['box3d_lidar'][i, 6].cpu().numpy()
                # getting bounding box score
                cls_score = pred_dict['scores'][i].cpu().numpy()
                
                # writing bounding box info
                f.write(f"{dimensions[0]} {dimensions[1]} {dimensions[2]} ")
                f.write(f"{location[0]} {location[1]} {location[2]} ")     
                f.write(f"{angle} ")       
                # writing classification score
                f.write(f"{cls_score}")


def save_anno_to_labelfile(path, anno):
    """
    Args: 
         - anno: prediction_annotation_file
    """
    idx = int2constString(anno['metadata']["image_idx"])
    # total dections in this frame
    num_detections = anno['dimensions'].shape[0]

    filepath = Path(path) / (idx + '.txt')
    with open(filepath, 'w') as f:

        for j in range(num_detections):

            if anno['name'][j] == 0:
                name = 'Car'
            else: 
                name = anno['name'][j]

            # initial information
            f.write(f"{name} {anno['truncated'][j]} {anno['occluded'][j]} {anno['alpha'][j]} ")

            # 2d bounding box
            f.write(f"{anno['bbox'][j][0]} {anno['bbox'][j][1]} {anno['bbox'][j][2]} {anno['bbox'][j][3]} ")

            # 3d bounding box dimensions
            # NOTE: Dimension should be written in form: height, width, length
            f.write(f"{anno['dimensions'][j][1]} {anno['dimensions'][j][2]} {anno['dimensions'][j][0]} ")
            #f.write(f"{anno['dimensions'][j][2]} {anno['dimensions'][j][1]} {anno['dimensions'][j][0]} ")

            # 3d bounding box location
            f.write(f"{anno['location'][j][0]} {anno['location'][j][1]} {anno['location'][j][2]} ")

            # 3d bounding box rotation
            f.write(f"{anno['rotation_y'][j]} ")
            
            # 3d bounding box prediction score
            f.write(f"{anno['score'][j]} \n")
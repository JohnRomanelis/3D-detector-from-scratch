import torch

def example_convert_to_torch(example, dtype=torch.float32, device=None):

    device = device or torch.device("cuda:0")

    example_torch = {}

    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weigths", "bev_map", "importance", "points"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)

        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        
        elif k in ["anchor_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)

        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib

        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
            
        else:
            example_torch[k] = v


    return example_torch
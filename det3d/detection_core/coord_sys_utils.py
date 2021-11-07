import torch
import numpy as np


def cyl2cart_torch(pc_cyl):
    
    x = pc_cyl[..., 0] * torch.cos(pc_cyl[..., 1])
    y = pc_cyl[..., 0] * torch.sin(pc_cyl[..., 1])
    z = pc_cyl[..., 2]
    x.unsqueeze_(-1)
    y.unsqueeze_(-1)
    z.unsqueeze_(-1)

    return torch.cat([x, y, z], dim=-1).reshape_as(pc_cyl)


def cyl2cart_numpy(pc_cyl):
    
    pc_cyl_shape = pc_cyl.shape

    x = pc_cyl[..., 0] * np.cos(pc_cyl[..., 1])
    y = pc_cyl[..., 0] * np.sin(pc_cyl[..., 1])
    z = pc_cyl[..., 2]
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    z = z[..., np.newaxis]

    return np.concatenate([x, y, z], axis=-1).reshape(pc_cyl_shape)


def cart2cyl_np(pc_cart):
    # coordinate transformation
    r = np.sqrt(np.power(pc_cart[..., 0], 2) + np.power(pc_cart[..., 1], 2))
    theta = np.arctan2(pc_cart[..., 1], pc_cart[..., 0])
    z = pc_cart[..., 2]
    # adding axis for concatenation
    r = r[..., np.newaxis]
    theta = theta[..., np.newaxis]
    z = z[..., np.newaxis]

    # returning concatenated result
    return np.concatenate([r, theta, z], axis=-1)


def cart2cyl_with_features_numpy(pc_cart):
    coors = pc_cart[..., :3]
    coors = cart2cyl_np(coors)

    pc_cart[..., :3] = coors

    return pc_cart

def cart2cyl_torch(pc_cart):
    # coordinate transformation
    r = torch.sqrt(torch.pow(pc_cart[..., 0], 2) + torch.pow(pc_cart[..., 1], 2))
    theta = torch.atan2(pc_cart[..., 1], pc_cart[..., 0])
    z = pc_cart[..., 2]
    # adding axis for concatenation
    r = r.unsqueeze_(-1)
    theta = theta.unsqueeze_(-1)
    z = z.unsqueeze_(-1)

    # returning concatenated result
    return torch.cat([r, theta, z], axis=-1)


def cart2cyl_with_features_torch(pc_cart):
    coors = pc_cart[..., :3]
    coors = cart2cyl_torch(coors)

    pc_cart[..., :3] = coors

    return pc_cart
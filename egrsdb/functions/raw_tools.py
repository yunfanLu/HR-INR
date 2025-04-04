import torch


def quad_bayes_to_bayes(raw):
    # raw shape: N, 1, H, W
    # change 1 2 raw
    new_raw = torch.zeros_like(raw)
    new_raw[:, :, 0::4, :] = raw[:, :, 0::4, :]
    new_raw[:, :, 1::4, :] = raw[:, :, 2::4, :]
    new_raw[:, :, 2::4, :] = raw[:, :, 1::4, :]
    new_raw[:, :, 3::4, :] = raw[:, :, 3::4, :]
    # change 3 4 col
    raw[:, :, :, 0::4] = new_raw[:, :, :, 0::4]
    raw[:, :, :, 1::4] = new_raw[:, :, :, 2::4]
    raw[:, :, :, 2::4] = new_raw[:, :, :, 1::4]
    raw[:, :, :, 3::4] = new_raw[:, :, :, 3::4]
    return raw

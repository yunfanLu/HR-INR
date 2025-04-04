#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/2 15:49
from os.path import join

import torch
from matplotlib import pyplot as plt

from egrsdb.functions.generate_2d_grid import generate_2d_grid_HW


def rolling_mask_to_image(title, mask, folder):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    im = ax.imshow(mask, cmap=plt.cm.PiYG, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im)
    plt.savefig(join(folder, f"{title}.png"))


def rolling_shutter_to_rolling_shutter_temporal_mask(
    rolling_start_time_begin,
    rolling_end_time_begin,
    rolling_start_time_end,
    rolling_end_time_end,
    high,
    time_bins,
    high_sample_count,
    time_sample_count,
):
    """
    :param rolling_start_time_begin: The start time of the rolling shutter.
    :param rolling_end_time_begin: The end time of the rolling shutter.
    :param rolling_start_time_end: The start time of the rolling shutter.
    :param rolling_end_time_end: The end time of the rolling shutter.
    :param high: The high of the image.
    :param time_bins: The number of time bins of optical flow.
    :param high_sample_count: The number of sample points in the high dimension.
    :param time_sample_count: The number of sample points in the time dimension.
    :return: [H, T] mask
    """
    A1 = rolling_start_time_begin - rolling_end_time_begin
    B1 = 1
    C1 = -rolling_start_time_begin
    A2 = rolling_start_time_end - rolling_end_time_end
    B2 = 1
    C2 = -rolling_start_time_end
    # Generate 2D grid
    grid = generate_2d_grid_HW(H=high, W=time_bins)
    direction_mask = torch.zeros([high, time_bins], dtype=torch.float32)
    for i in range(high_sample_count):
        for j in range(time_sample_count):
            center_grid = grid.clone()
            X = center_grid[0, :, :]
            Y = center_grid[1, :, :]
            dx = 1.0 * i / high_sample_count
            X = X + dx
            dy = 1.0 * j / time_sample_count
            Y = Y + dy
            X = X / high
            Y = Y / time_bins
            # rolling select mask
            tmp_rolling_mask1 = A1 * X + B1 * Y + C1
            tmp_rolling_mask2 = A2 * X + B2 * Y + C2
            # intersecting
            tmp_mask = tmp_rolling_mask1 * tmp_rolling_mask2
            mask = torch.zeros([high, time_bins], dtype=torch.float32)
            mask[tmp_mask <= 0] = 1.0
            direction_mask += mask
    direction_mask /= high_sample_count * time_sample_count
    # direction, if left to right the direction is positive, else the direction is negative.
    left_to_right = 1 if rolling_start_time_begin < rolling_start_time_end else -1
    direction_mask *= left_to_right
    return direction_mask


def rolling_shutter_to_global_shutter_temporal_mask(
    rolling_start_time,
    rolling_end_time,
    global_time,
    high,
    time_bins,
    high_sample_count,
    time_sample_count,
):
    """
    generate an optical flow mask for rolling shutter to global shutter.
    the left to right direction is positive.
    :param rolling_start_time: The start time of the rolling shutter.
    :param rolling_end_time: The end time of the rolling shutter.
    :param global_time: The global timestamp.
    :param high: The high of the image.
    :param time_bins: The number of time bins of optical flow.
    :param high_sample_count: The number of sample points in the high dimension.
    :param time_sample_count: The number of sample points in the time dimension.
    :return: [H, T] mask
    """
    assert rolling_start_time < rolling_end_time
    assert rolling_start_time >= 0 and rolling_end_time <= 1
    assert 0 <= global_time <= 1
    # The rolling shutter dividing line.
    # This line is A*x + B*y + C = 0
    A = rolling_start_time - rolling_end_time
    B = 1
    C = -rolling_start_time
    # Generate 2D grid
    grid = generate_2d_grid_HW(H=high, W=time_bins)
    direction_mask = torch.zeros([high, time_bins], dtype=torch.float32)
    for i in range(high_sample_count):
        for j in range(time_sample_count):
            center_grid = grid.clone()
            X = center_grid[0, :, :]
            Y = center_grid[1, :, :]
            dx = 1.0 * i / high_sample_count
            X = X + dx
            dy = 1.0 * j / time_sample_count
            Y = Y + dy
            X = X / high
            Y = Y / time_bins
            # rolling select mask
            tmp_rolling_mask = A * X + B * Y + C
            tmp_global_mask = Y - global_time
            # intersecting
            tmp_mask = tmp_rolling_mask * tmp_global_mask
            mask = torch.zeros([high, time_bins], dtype=torch.float32)
            mask[tmp_mask <= 0] = 1.0
            mask[Y > global_time] *= -1
            direction_mask += mask
    direction_mask = direction_mask / (high_sample_count * time_sample_count)
    return direction_mask

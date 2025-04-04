import numba
import numpy as np
from absl.logging import info

DEBUG = False


def _render(x, y, p, shape):
    events = np.zeros(shape=shape, dtype=np.float32)
    events[y, x] = p
    return events


def event_stream_to_temporal_pyramid_representation(
    events, pyramid_level, pyramid_moments, reduction_factor, resolution
):
    """
    events: [t,x,y,p], p = 0 or 1
    return [L,M,H,W]
    """
    H, W = resolution

    #
    event_time_pyramid_voxel = np.zeros(shape=[pyramid_level, pyramid_moments, H, W], dtype=np.float32)

    if events.shape[0] == 0:
        return event_time_pyramid_voxel
    #
    begin_time = events[:, 0].min()
    end_time = events[:, 0].max()
    during_time = end_time - begin_time

    time_start_end_list = [[0, 1]]
    for i in range(pyramid_level):
        l, r = time_start_end_list[-1]
        during = r - l
        deta = (during - during / reduction_factor) / 2
        l = l + deta
        r = r - deta
        time_start_end_list.append([l, r])

    if DEBUG:
        info(f"begin_time: {begin_time}")
        info(f"end_time: {end_time}")
        info(f"during_time: {during_time}")
        info(f"time_start_end_list: {time_start_end_list}")

    for i in range(pyramid_level):
        l, r = time_start_end_list[i]
        l = l * during_time + begin_time
        r = r * during_time + begin_time
        moment_during_time = r - l

        for j in range(pyramid_moments):
            m_t_l = l + moment_during_time * j / pyramid_moments
            m_t_r = l + moment_during_time * (j + 1) / pyramid_moments
            left_index = np.searchsorted(events[:, 0], m_t_l, side="left")
            right_index = np.searchsorted(events[:, 0], m_t_r, side="right")
            li, ri = left_index, right_index

            x, y, p = events[li:ri, 1], events[li:ri, 2], events[li:ri, 3]
            x = x.astype(np.int32)
            y = y.astype(np.int32)
            event_voxel_grid = _render(x=x, y=y, p=p, shape=resolution)
            event_time_pyramid_voxel[i, j] = event_voxel_grid
    return event_time_pyramid_voxel

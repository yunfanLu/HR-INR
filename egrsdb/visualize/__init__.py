from egrsdb.visualize.demosaic_hybridevs_visualization import DemosaicHybridevsBatchVisualization
from egrsdb.visualize.rolling_shutter_visualization import VisualizationRollingShutter
from egrsdb.visualize.vfi_sr_batch_visualization import VFISRBatchVisualization


def get_visulization(config):
    if config.NAME == "rs-vis":
        return VisualizationRollingShutter(config)
    elif config.NAME == "vfi-sr-batch-vis":
        return VFISRBatchVisualization(config)
    elif config.NAME == "demosaic-vis":
        return DemosaicHybridevsBatchVisualization(config)
    else:
        raise NotImplementedError(f"Visualization {config.NAME} is not implemented.")

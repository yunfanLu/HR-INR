from torch.nn.modules.loss import _Loss

from egrsdb.datasets.basic_batch import VFI_SR_BATCH


class VFISRReconstructedMetric(_Loss):
    def __init__(self, metric, remove_input_frames, to_gray=False):
        super(VFISRReconstructedMetric, self).__init__()
        self.metric = metric
        self.remove_input_frames = remove_input_frames
        self.to_gray = to_gray

    def forward(self, batch):
        # gt frames
        gt_frames = batch[VFI_SR_BATCH.HFR_HR_FRAMES]
        # pred frames
        pred_frames = batch[VFI_SR_BATCH.HFR_HR_FRAMES_PRED]
        B, N, C, H, W = gt_frames.shape
        count = 0
        metric = 0
        for i in range(N):
            if self.remove_input_frames and (i == 0 or i == N - 1):
                continue
            gt = gt_frames[:, i, :, :, :]
            pred = pred_frames[:, i, :, :, :]
            if self.to_gray:
                gt = gt.mean(dim=1, keepdim=True)
                pred = pred.mean(dim=1, keepdim=True)
            metric = metric + self.metric(gt, pred).float()
            count = count + 1
        return metric / count

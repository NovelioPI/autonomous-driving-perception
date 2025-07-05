import numpy as np

class ConfusionMatrix:
    def __init__(self, num_classes, ignore_label=255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.M = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, gt, pred):
        """
        Efficiently update confusion matrix from arrays.
        Args:
            gt (ndarray): Ground truth labels (flattened or 1D).
            pred (ndarray): Predicted labels (same shape as gt).
        """
        gt = np.asarray(gt).flatten()
        pred = np.asarray(pred).flatten()

        # Mask for valid ground truth labels
        mask = (gt != self.ignore_label) & (gt < self.num_classes) & (pred < self.num_classes)
        gt = gt[mask]
        pred = pred[mask]

        cm = np.bincount(self.num_classes * gt + pred, minlength=self.num_classes**2)
        self.M += cm.reshape(self.num_classes, self.num_classes)
        
    def add_batch(self, gt_batch, pred_batch):
        """
        Efficiently update confusion matrix from batches of arrays.
        Args:
            gt_batch (list of ndarray): List of ground truth labels.
            pred_batch (list of ndarray): List of predicted labels.
        """
        for gt, pred in zip(gt_batch, pred_batch):
            self.add(gt, pred)

    def addM(self, matrix):
        self.M += matrix

    def reset(self):
        self.M = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def recall(self):
        # Per-class recall: TP / (TP + FN)
        tp = np.diag(self.M)
        fn = self.M.sum(axis=1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = tp / (tp + fn)
        return np.nanmean(recall)

    def accuracy(self):
        # Per-class accuracy: TP / (TP + FP)
        tp = np.diag(self.M)
        fp = self.M.sum(axis=0) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = tp / (tp + fp)
        return np.nanmean(accuracy)

    def jaccard(self):
        # IoU = TP / (TP + FP + FN)
        tp = np.diag(self.M)
        fp = self.M.sum(axis=0) - tp
        fn = self.M.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / denom
        valid = denom > 0
        mean_iou = np.nanmean(iou[valid])
        return mean_iou, iou.tolist(), self.M.copy()

def get_iou(data_list, class_num, save_path=None):
    """ 
    Args:
        data_list: list of tuples (gt, pred) [arrays or tensors]
        class_num: number of classes
        save_path: optional, where to save output
    Returns:
        mean IoU, per-class IoU
    """
    cm = ConfusionMatrix(class_num)
    for gt, pred in data_list:
        cm.add(gt, pred)
    mean_iou, per_class_iou, _ = cm.jaccard()

    return mean_iou, per_class_iou

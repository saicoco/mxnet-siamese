# coding=utf-8

import mxnet as mx
import numpy as np

class siamise_metric(mx.metric.EvalMetric):

    def __init__(self, name='siamise_metric'):
        super(siamise_metric, self).__init__(name=name)

    def update(self, label, pred):
        preds = pred[0]
        labels = label[0]
        preds_label = preds.asnumpy().ravel()
        labels = labels.asnumpy().ravel()
        preds_label[preds_label >= 0.5] = 1.0
        preds_label[preds_label < 0.5] = 0
        preds_label = 1.0 - preds_label
        self.sum_metric += (labels==preds_label).sum()
        self.num_inst += len(labels)


# coding=utf-8
import mxnet as mx
class siamise_metric(mx.metric.EvalMetric):

    def __init__(self, name='siamise_acc'):
        super(siamise_metric, self).__init__(name=name)

    def update(self, label, pred):
        preds = pred[0]
        labels = label[0]
        preds_label = preds.asnumpy().ravel()
        labels = labels.asnumpy().ravel()
        self.sum_metric += labels[preds_label < 0.5].sum() + len(preds_label >= 0.5).sum()
        self.num_inst += len(labels)

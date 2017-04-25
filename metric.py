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
        self.sum_metric += labels[preds_label < 0.5].sum() + len(labels[preds_label >= 0.5]) - labels[preds_label >= 0.5].sum()
        self.num_inst += len(labels)

class contrastive_loss(mx.metric.EvalMetric):
    def __init__(self, name='contrastive_loss'):
        super(contrastive_loss, self).__init__(name=name)
    
    def update(self, label, pred):
        loss = pred[1].asnumpy()
        self.sum_metric += loss
        self.num_inst += len(loss)

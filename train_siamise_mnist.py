# coding=utf-8

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
from siamise_sym import *
from dataloader import PairDataIter
from metric import siamise_metric, contrastive_loss

def main():
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of batch')
    parser.add_argument('--epoches', type=int, default=50,
                        help='epoches of train stage')
    parser.add_argument('--gpus', type=str, default=None, help="context device")
    args = parser.parse_args()
    siamise_sym = compose_sym()
    ctx = mx.cpu() if not args.gpus else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    mod = mx.mod.Module(
        symbol=siamise_sym,
        data_names=('pos_data', 'neg_data'),
        label_names=('sim_label',),
        context=mx.cpu()
    )

    train_iter = PairDataIter(batch_size=args.batch_size, mode='train')
    val_iter = PairDataIter(batch_size=args.batch_size, mode='val')
    siametric = siamise_metric()
    contra_loss = contrastive_loss()
    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(siametric)
    eval_metric.add(contra_loss)
    mod.fit(
        train_data=train_iter,
        eval_data=val_iter,
        eval_metric=eval_metric,
        batch_end_callback=mx.callback.Speedometer(args.batch_size, frequent=50),
        num_epoch=args.epoches,
        optimizer='sgd',
        optimizer_params=(('learning_rate', 0.1),),
        validation_metric=eval_metric
    )

if __name__=='__main__':
    main()
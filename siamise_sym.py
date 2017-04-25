# codeing=utf-8

'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
'''

import mxnet as mx

def mlp(data, fc_weights, fc_bias):
    data = mx.sym.Flatten(data=data)

    num_hiddens = [128, 128, 128]
    for i in xrange(3):
        data = mx.symbol.FullyConnected(data = data, weight=fc_weights[i], bias=fc_bias[i], name='fc'+str(i), num_hidden=num_hiddens[i])
        data = mx.symbol.Activation(data=data, act_type='relu', name='fc'+str(i))   
    fc = mx.symbol.L2Normalization(data=data, name='l2')
    return fc

def lenet(data, conv_weights, conv_bias, fc_weights, fc_bias):
    convi = data
    num_filters = [64, 128, 512]
    kernels = [5, 3, 3]
    for i in range(len(num_filters)):
        convi = mx.sym.Convolution(data=convi, kernel=(kernels[i], kernels[i]), stride=(1, 1), \
                                    num_filter=num_filters[i], weight=conv_weights[i], bias=conv_bias[i])
        convi = mx.sym.Activation(convi, act_type='relu')
        convi = mx.sym.Pooling(data=convi, kernel=(2, 2), pool_type='max')
    flatten = mx.sym.Flatten(convi)

    num_hiddens = [512, 256]
    for i in xrange(len(num_hiddens)):
        data = mx.symbol.FullyConnected(data = data, weight=fc_weights[i], bias=fc_bias[i], name='fc'+str(i), num_hidden=num_hiddens[i])
        data = mx.symbol.Activation(data=data, act_type='relu', name='fc'+str(i))   
    fc = mx.symbol.L2Normalization(data=data, name='l2')
    return fc

def compose_sym(margin=1.0):
    
    pos_data = mx.symbol.Variable('pos_data')
    neg_data = mx.symbol.Variable('neg_data')
    label = mx.symbol.Variable('sim_label')
    
    fc_weights = []
    fc_bias = []
    conv_weights = []
    conv_bias = []

    for i in xrange(3):
        conv_weights.append(mx.sym.Variable('conv'+str(i) + 'weight'))
        conv_bias.append(mx.sym.Variable('conv'+str(i) + 'bias'))
    for i in xrange(2):
        fc_weights.append(mx.sym.Variable('fc'+str(i) + 'weight'))
        fc_bias.append(mx.sym.Variable('fc'+str(i) + 'bias'))
    
    pos_out = lenet(pos_data, conv_weights, conv_bias, fc_weights, fc_bias)
    neg_out = lenet(neg_data, conv_weights, conv_bias, fc_weights, fc_bias)

    pred = mx.sym.sqrt(mx.sym.sum(mx.sym.square(pos_out - neg_out), axis=1, keepdims=True))
    loss = mx.sym.mean(label * mx.sym.square(pred) + (1 - label) * mx.sym.square(mx.sym.maximum(margin - pred, 0)))
    contrative_loss = mx.sym.MakeLoss(loss, name='loss')
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(pred, name='pred'), contrative_loss])
    return pred_loss


if __name__ == "__main__":
    import numpy as np
    siamise_sym = compose_sym()
    exe = siamise_sym.simple_bind(pos_data=(10, 784), neg_data=(10, 784), sim_label=(10,1), ctx=mx.cpu())
    pos_data = mx.nd.array(np.random.random(size=(10, 784)))
    neg_data = mx.nd.array(np.random.random(size=(10, 784)))
    labels = mx.nd.array(np.random.random(size=(10,1)))
  
    exe.forward(pos_data=pos_data, neg_data=neg_data, sim_label=labels, is_train=True)
    exe.backward()
    print exe.outputs[0].asnumpy(), exe.outputs[1].asnumpy()





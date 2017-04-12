# mxnet-siamise
siamise networks[^1]  

Here use simple mlp as forward networks, you can replace it with CNN or anything complicate netowrks.

[^1] ["Dimensionality Reduction by Learning an Invariant Mapping"](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
    
### Usage  

```
python train_siamise_mnist.py --batch_size 128 --epoches 20 --gpu 0,1
```



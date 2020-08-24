# SymmetryNet
SymmetryNet: Learning to Predict Reflectional and Rotational Symmetries of 3D Shapes from Single-View RGB-D Images

ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2020)

Created by Yifei Shi, Junwen Huang, Hongjia Zhang, Xin Xu, Szymon Rusinkiewicz and Kai Xu

* **tools**: the training scripts and evaluation scripts
* **lib**: the core Python library for networks and loss
* **datasets**: the dataloader and training/testing lists

## Environments
pytorch>=0.4.1
python >=3.6

## Loss:
> center-loss + ref-ctp-loss + ref-foot-loss + rot-foot-loss + num-loss + mode-loss + ref-co-loss + rot-co-loss

* **ref-ctp-loss** : the counterpart error,i.e. ctp-dis/ctp-len
* **num-loss** : optmal assigment loss
* **mode-loss** : classification of symmetry types, ref-only(0)? rot-only(1)? both(2)?
* **ref-co-loss** : for reflectional symmetry, the distance from input point to its counterpart point is 2 times as large as the distance from input point to its foot point on the symmetry, i.e. ctp-dis = 2*foot-pt-dis
* **rot-co-loss**: for rotational symmetry, the vector from an input point to its foot point is perpendicular to the vector from center to the foot point,i.e. center-to-foot ⊥ pt-to-foot

## Traing
To simply train the network with the default parameter on shapenet dataset, run<br>
```
python tools/train.py
```

## Evaluation
To evaluate the model with our metric on shapenet, for reflectional symmetry, run<br>

```
python tools/evaluation/eval_ref_shapenet.py
```

for rotational symmetry, run<br>

```
python tools/evaluation/eval_rot_shapenet.py
```




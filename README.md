# SymmetryNet
SymmetryNet: Learning to Predict Reflectional and Rotational Symmetries of 3D Shapes from Single-View RGB-D Images
loss:
loss_all.py : (all loss)
center-loss + ref-ctp-loss + ref-foot-loss + rot-foot-loss + num-loss + mode-loss + ref-co-loss + rot-co-loss

ref-ctp-loss : ctp-error/ctp-len
num-loss : optmal assigment loss
mode-loss : ref-only(0)? rot-only(1)? both(2)?
ref-co-loss : ctp-dis = 2*foot-pt-distance
rot-co-loss: center-to-foot ⊥ pt-to-foot

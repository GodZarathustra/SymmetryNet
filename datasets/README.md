## Data format

### Network inputs:
**ShapeNet**
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-color.png: rendered color image
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-depth.exr: rendered depth image
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-k.txt: intrinsic parameters
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-rt.txt: extrinsic parameters of the cropped image
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-color-crop.png: cropped color image
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-depth-crop.png: cropped depth image
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-color-crop-occlusion.png: cropped color image with occlusion
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-depth-crop-occlusion.png: cropped depth image with occlusion
- SUBSET/CLASS_ID/INSTANCE_ID/VIEW_ID-k-crop.txt: intrinsic parameters of the cropped image
- SUBSET/CLASS_ID/INSTANCE_ID/info.txt: shapenet instance id, rendering details

The scale factor of depth image is 100.

**ScanNet**
Network inputs:
- SUBSET/SCENE_ID/VIEW_ID-color.jpg: color image
- SUBSET/CLASS_ID/VIEW_ID-depth.png: depth image
- SUBSET/CLASS_ID/VIEW_ID-label.png: object segmentation of the depth image
- SUBSET/CLASS_ID/VIEW_ID-meta.mat: intrinsic, extrinsic parameters, scale factor, object center, shapenet v2 instance id, object center in image space, scale factor of depth image

**YCB**
Network inputs:
We use the format as [PoseCNN][1].


### Symmetry annotations:
We provide instance-level symmetry annotation. The format is:
~~~~ 
object_center_x object_center_y object_center_z
ref_sym1_normal_x ref_sym1_normal_y ref_sym1_normal_z
ref_sym2_normal_x ref_sym2_normal_y ref_sym2_normal_z
ref_sym3_normal_x ref_sym3_normal_y ref_sym3_normal_z
rot_sym_direction_x rot_sym_direction_y rot_sym_direction_z
~~~~

[1]:  https://rse-lab.cs.washington.edu/projects/posecnn/ "PoseCNN"
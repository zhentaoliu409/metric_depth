# Pothole Repair 
![1](/assets/pothole-repair.png)
## Environment  
matplotlib: 3.9.2  
opencv-python: 4.10.0  
open3d: 0.18.0  
torch: 1.13.1  
ultralytics 8.3.19  
torchvision: 0.14.1  
CUDA version: 11.7   
Python 3.10.15  
![2](/assets/4.png)
## Usage  

###
Download the model like [YOLO11n.pt and YOLO11n-seg.pt](https://github.com/ultralytics/ultralytics) and put it into the yolo directory. 

### Train the model and try to get inference through the trained model
```bash
python yolo/train.py
```
```bash
python yolo/inference.py --model-path yolo/runs/segment/train/weights/best.pt --image-path "your image path"
```

###
Download the weights file like [depth_anything_v2_metric_vkitti_vitl.pth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main) and put it into the checkpoints directory.  

### Generate depth map from 'inputs' to 'outputs' directory
```bash
python run.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 80 --img-path './inputs' --outdir './outputs'
```

### Save the depth map as a point cloud file
```bash
python depth_to_pointcloud.py  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 20 --img-path './inputs' --outdir './outputs'
```

### Generate a point cloud of the original view
```bash
python 3dpoints.py
```

## Refernces
The URL of the original article are [YOLO11](https://github.com/ultralytics/ultralytics) and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main). The dataset is derived from [roadvis-segmentation_dataset](https://universe.roboflow.com/sankritya-rai-cldft/roadvis-segmentation)  

```bash
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

```bash
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

```bash
@misc{
roadvis-segmentation_dataset,
title = { RoadVis Segmentation  Dataset },
type = { Open Source Dataset },
author = { Sankritya Rai },
howpublished = { \url{ https://universe.roboflow.com/sankritya-rai-cldft/roadvis-segmentation } },
url = { https://universe.roboflow.com/sankritya-rai-cldft/roadvis-segmentation },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2023 },
month = { apr },
note = { visited on 2025-01-13 },
}
```

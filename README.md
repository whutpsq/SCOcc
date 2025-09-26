# Multi-Scale Spatial-Channel Attention Fusion Network for 3D Occupancy Prediction


<div align="left">
  <img src="figs/performance_scocc.png"  width="400px" />
</div><br/>
The performance of SCOcc comparison with other state-of-the-art methods.


<!-- ## Introduction -->
This repository is an official implementation of SCOcc.

<div align="left">
  <img src="figs/pipeline.png" width="600px"/>
</div><br/>

## TODO Lists
* Release the visualization code.


## Main Results
| Config                                                                                         | Backbone | Input <br/>Size | mIoU  | Model                                                                                                      |
|:-----------------------------------------------------------------------------------------------|:--------:|:---------------:|:-----:|:-----------------------------------------------------------------------------------------------------------|
| [**SCOcc-4D-Stereo (6f)**](projects/configs/scocc/scocc-r50-4d-stereo.py)                      |   R50    |     256x704     | 40.51 | [**scocc-r50-6f**](https://drive.google.com/file/d/1bISLcGrgBY_lIdtlGvaiPcBuCwZAEOCZ/view?usp=drive_link)  | 
| [**SCOcc-4D-Stereo (4f)**](projects/configs/scocc/scocc-stbase-4d-stereo-512x1408_4x4_2e-4.py) |  Swin-B  |    512x1408     | 45.01 | [**scocc-swin-4f**](https://drive.google.com/file/d/1JscU3Vg0e_UmsccRKMrs4ttrjgKYAOuc/view?usp=drive_link) | 


## Get Started
1. [Environment Setup](doc/install.md)
2. [Model Training](doc/model_training.md)
3. [Visualization](doc/visualization.md)
<div>
  <img src="figs/visualization.png" width="600px"/>
</div><br/>

<div style="display: flex; gap: 10px;">
  <div>
    <img src="figs/vis_front.gif" width="400px" /> 
  </div>

  <div>
    <img src="figs/vis_bev.gif" width="400px" />
  </div>
</div>


## Acknowledgement
Many thanks to the following open-source projects:
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
* [RenderOcc](https://github.com/pmj110119/RenderOcc.git)
* [PanoOcc](https://github.com/Robertwyq/PanoOcc.git)
* [FlashOCC](https://github.com/Yzichen/FlashOCC.git)

[//]: # (## Bibtex)

[//]: # (If this work is helpful for your research, please consider citing the following BibTeX entry.)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (```)

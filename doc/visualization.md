
# SCOcc
```shell
# step 1. generate result 
bash tools/dist_test.sh projects/configs/scocc/scocc-r50.py ckpts/scocc-r50-256x704.pth 4 --eval map --eval-options show_dir=work_dirs/scocc_r50/results
# step 2. visualization
python tools/analysis_tools/vis_occ.py work_dirs/scocc_r50/results/ --root_path ./data/nuscenes --save_path ./vis
```



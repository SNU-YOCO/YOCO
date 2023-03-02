# YOCO : You Only Cook Once

<img src="teaser/delicious.png" width="50%"> <img src="teaser/title.png" width="35%">

<img src="teaser/raw_crop.gif" width="30%"><img src="teaser/cooked_crop.gif" width="30%"><img src="teaser/overcooked.gif" width="20%">

## Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors

<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch implementation of our paper: <br>
**Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors**<br>
[Yen-Cheng Liu](https://ycliu93.github.io/), [Chih-Yao Ma](https://chihyaoma.github.io/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/)<br>
The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2022 <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.pdf)] [[Project](https://ycliu93.github.io/projects/unbiasedteacher2.html)]

<p align="center">
<img src="teaser/teaser_utv2.png" width="80%">
</p>

## Training

### FCOS

- Train Unbiased Teacher v2 under 40% COCO-supervision (adjust SUP_PERCENT for different ratio )

```shell
python train_net_yoco.py\
      --num-gpus 1 \
      --config configs/FCOS/coco-standard/yoco_fcos_R_50_ut2_run0.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4  \
       SOLVER.MAX_ITER 50000 SEMISUPNET.BURN_UP_STEP 20000  \
       TEST.EVAL_PERIOD 500 DATALOADER.SUP_PERCENT 40.0
```

## Resume the training

```shell
python train_net_yoco.py \
      --resume \
      --num-gpus 1 \
      --config configs/FCOS/coco-standard/yoco_fcos_R_50_ut2_run0.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4  \
       SOLVER.MAX_ITER 50000 SEMISUPNET.BURN_UP_STEP 20000  \
       TEST.EVAL_PERIOD 500 DATALOADER.SUP_PERCENT 40.0  \
       MODEL.WEIGHTS <weight_file_name>.pth
```

## Inference Only

- Adjust INFERENCE_TH_TEST for different threshold

```shell
python train_net.py \
      --test-only \
      --num-gpus 1 \
      --config configs/FCOS/coco-standard/yoco_fcos_R_50_ut2_run0.yaml \
      --output_dir <output_file_directory> \
      --video_input <input_file_path> \
      MODEL.WEIGHTS <weight_file_name>.pth \
      MODEL.FCOS.INFERENCE_TH_TEST 0.4
```

## License

This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.

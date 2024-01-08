# ObjectFormer for Image Manipulation Detection and Localization

Official PyTorch implementation of [ObjectFormer for Image Manipulation Detection and Localization](https://arxiv.org/abs/2203.14681), CVPR 2022.

## Training and Evaluation

Please first download the imagenet-pretrained model from this [Google Drive](https://drive.google.com/file/d/1I-H58ldwwvoKD1GX0jdlNJGA9hqgYNKy/view?usp=sharing) link.

Then for training, you could use:

```
python tools/run.py --cfg configs/objectformer_bs24_lr2.5e-4.yaml
```

while for evaluation, you could set TRAIN.ENABLE to False. For a better peformance on Pixel F1, you should adjust the TEST.THRES (0.5 by default) on each testing dataset.

Note that we only release the checkpoints trained on the publicly available dataset ([CASIAV2](https://drive.google.com/file/d/1vd7o7JI-_EukyplskeuSWuham4iCWuTq/view?usp=sharing) and [IMD20](https://drive.google.com/file/d/1IWQvoMl9iaefCLbF5gXVbdB2ICfSXLPP/view?usp=sharing)). For a fair comparison with our method, please finetune it with your data.

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{wang2022objectformer,
  title={Objectformer for image manipulation detection and localization},
  author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Han, Xintong and Shrivastava, Abhinav and Lim, Ser-Nam and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

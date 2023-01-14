# Facial_Attributes_Obfuscation

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

[ACM MM21] Official Code: 

Jingzhi Li, Lutong Han, Ruoyu Chen, Hua Zhang, Bing Han, Lili Wang, Xiaochun Cao:
**Identity-Preserving Face Anonymization via Adaptively Facial Attributes Obfuscation**. ACM Multimedia 2021: 3891-3899

![](./figure/framework.png)

> Note: We received a lot of requests from researchers about the code, so we decided to open source some of the collated code.

## Preparation

Please refer to [model_zoo.md](./pretrained/model_zoo.md) to download the pre-trained model to the folder [pretrained](./pretrained/).

Please align all your face images before. If you want to use vggface net, try `mode=vggface`, if you want to use arcface net, try `mode=arcface`. See [tutorial_facealignment.ipynb](tutorial_facealignment.ipynb).

## Stage 1, Identity-aware region discovery

run command:

```
sh stage1.sh
```

than, you can view the results in the saved fold (default is [results](results)). The facial attribute importance sort is writen in a json file like:

```json
{
    "0001_01-0.jpg":{
        "ID":0,
        "ImagePath":"./align_vgg/0001_01-0.jpg",
        "ScoreSort":{
            "nose":0.8627879597363718,
            "eyes":0.8358434626789568,
            "mouth":0.7089767999216268,
            "eyebrows":0.6859111387108845,
            "skin":0.6761247350748167,
            "hair":0.2833600649095848
        },
        "GradCAMPath":"results/vgg_mode/Img/0001_01-0.jpg/0001_01-0-gradcam.jpg",
        "FacePartScorePath":"results/vgg_mode/Img/0001_01-0.jpg/0001_01-0-part-score.jpg"
    },
    ...
}
```

## Stage 2, Identity-aware face obfuscation

After obtaining the most important attribute region, modify according to this region.

In this part, you can directly use the official stargan v2, and combine the results obtained in the first stage for training and testing.

[https://github.com/clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)

If you have any questions about this part of the operation, please contact me.

## Acknowledgement

```bibtex
@inproceedings{li2021identity,
  title={Identity-Preserving Face Anonymization via Adaptively Facial Attributes Obfuscation},
  author={Li, Jingzhi and Han, Lutong and Chen, Ruoyu and Zhang, Hua and Han, Bing and Wang, Lili and Cao, Xiaochun},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3891--3899},
  year={2021}
}
```
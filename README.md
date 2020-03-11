# FaceBoxes in PyTorch

严安

[PyTorch](https://pytorch.org/) 实现的 [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234)

## 安装

1. PyTorch版本大雨1.0.0

2. 编译nms:

```Shell
./make.sh
```

代码基于Python3+实现

## 训练

1. 将图片和voc格式的xml文件分别放到/data/WIDER_FACE/images和/data/WIDER_FACE/annotations下（标注数据最好是正方形，anchor设定的是1:1）
2. 运行/data/WIDER_FACE/select_samples.py文件，将数据分成训练集和数据集

```python
python select_samples.py -i ./images -s 0.1 -t ./test -c 1
```

3. 生成文件列表文件

```python
python img_list.py --imgPath ./images
```

4. 根据任务修改类别和具体的类别名称

修改data/wider_voc.py WIDER_CLASSES = ( '__background__', 'face','face_mask')

和train.py目录下的classes

5. 训练模型

```Shell
python train.py
```

6. 只测试人脸一类运行python test.py，否则运行python test2.py

## 评估

参考目标检测评估程序

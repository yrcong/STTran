# Spatial-Temporal Transformer for Dynamic Scene Graph Generation
Pytorch Implementation of our paper [Spatial-Temporal Transformer for Dynamic Scene Graph Generation](https://arxiv.org/abs/2107.12309) accepted by **ICCV2021**. We propose a Transformer-based model **STTran** to generate dynamic scene graphs of the given video. **STTran** can detect the visual relationships in each frame.

**The introduction video is available now:** [https://youtu.be/gKpnRU8btLg](https://youtu.be/6D3ExjQpbjQ)

![GitHub Logo](/data/framework.png)

**About the code**
We run the code on a single RTX2080ti for both training and testing. We borrowed some code from [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch) and [Zellers' repository](https://github.com/rowanz/neural-motifs).

## Requirements
- python=3.6
- pytorch=1.1
- scipy=1.1.0
- cypthon
- dill
- easydict
- h5py
- opencv
- pandas
- tqdm
- yaml

## Usage
We use python=3.6, pytorch=1.1 and torchvision=0.3 in our code. First, clone the repository:
```
git clone https://github.com/yrcong/STTran.git
```
We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```

## Train
You can train the **STTran** with train.py. We trained the model on a RTX 2080ti:
+ For PredCLS: 
```
python train.py -mode predcls -datasize large -data_path $DATAPATH 
```
+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -data_path $DATAPATH 
```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -data_path $DATAPATH 
```

## Evaluation
You can evaluate the **STTran** with test.py.
+ For PredCLS ([trained Model](https://drive.google.com/file/d/1Sk5qFLWTZmwr63fHpy_C7oIxZSQU16vU/view?usp=sharing)): 
```
python test.py -m predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGCLS ([trained Model](https://drive.google.com/file/d/1ZbJ7JkTEVM9mCI-9e5bCo6uDlKbWttgH/view?usp=sharing)): : 
```
python test.py -m sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGDET ([trained Model](https://drive.google.com/file/d/1dBE90bQaXB-xogRdyAJa2A5S8RwYvjPp/view?usp=sharing)): : 
```
python test.py -m sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH
```

## Citation
If our work is helpful for your research, please cite our publication:
```
@inproceedings{cong2021spatial,
  title={Spatial-Temporal Transformer for Dynamic Scene Graph Generation},
  author={Cong, Yuren and Liao, Wentong and Ackermann, Hanno and Rosenhahn, Bodo and Yang, Michael Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16372--16382},
  year={2021}
}
```
## Help 
When you have any question/idea about the code/paper. Please comment in Github or send us Email. We will reply as soon as possible.

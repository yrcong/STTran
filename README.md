# Spatial-Temporal Transformer for Dynamic Scene Graph Generation
Pytorch Implementation of our paper [Spatial-Temporal Transformer for Dynamic Scene Graph Generation](https://arxiv.org/abs/2107.12309) accepted by **ICCV2021**. We propose a Transformer-based model **STTran** to generate dynamic scene graphs of the given video. **STTran** can detect the visual relationships in each frame.

![GitHub Logo](/data/framework.png)

**About the code**
We run the code on a single RTX2080ti for both training and testing. We borrowed some code from [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch) and [Zellers' repository](https://github.com/rowanz/neural-motifs).

## Usage
We use python=3.6, pytorch=1.1 and torchvision=0.3 in our code. First, clone the repository:
```
git clone https://github.com/yrcong/STTran.git
```
We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py
cd ..
cd fpn/box_intersections_cpu
python setup.py
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch

## Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
## Train



## Citation
If our work is helpful for your research, please cite our publication:
```
@article{cong2021spatial,
  title={Spatial-Temporal Transformer for Dynamic Scene Graph Generation},
  author={Cong, Yuren and Liao, Wentong and Ackermann, Hanno and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={arXiv preprint arXiv:2107.12309},
  year={2021}
}
```

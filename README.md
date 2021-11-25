# HelixTransformer
This repository contains the reference Pytorch source code for the following paper:

HelixTransformer: A Cross-attention Relation Transformer for Few-shot Learning(need to add url)

If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@InProceedings{
}
```

## Code environment
This code requires Pytorch 1.7.1 and torchvision 0.8.2 or higher with cuda support. It has been tested on Ubuntu 18.04. 

You can create a conda environment with the correct dependencies using the following command lines:
```
conda env create -f environment.yml
conda activate (need to add env name)
```

## Setting up data
You must first specify the value of `data_path` in `config.yml`. This should be the absolute path of the folder where you plan to store all the data.

The following datasets are used in our paper: 
- CUB_200_2011 \[[Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)\]
- FGVC-Aircraft \[[Dataset Page](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)\]
- Stanford Dogs \[[Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)\]
- Stanford Cars \[[Dataset Page](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)\]
- NABirds \[[Dateset Page](https://dl.allaboutbirds.org/nabirds)\]
- tiered-ImageNet \[[Dataset Page](https://github.com/renmengye/few-shot-ssl-public)\]

After setting up few-shot datasets following the steps above, the following folders will exist in your `data_path`:
- `CUB_crop`: 100/50/50 classes for train/validation/test, using bounding-box cropped images as input
- `Aircraft`: 60/15/55 classes for train/validation/test
- `StanfordDogs`: 70/20/30 classes for train/validation/test
- `StanfordCars`: 130/17/49 classes for train/validation/test
- `NABirds`: 350/66/139 classes for train/validation/test
- `tiered-ImageNet`: 351/91/160 classes for train/validation/test, images are 84x84

## Train and test
For fine-grained few-shot classification, we provide the training and inference code for both HelixTransformer and our Relation Network baseline, as they appear in the paper. 

Training a model can be simply divided into two stages: 
- Stage one: Pretraining backbone, run the following command line
```
python train_classifier.py
```
datasets and backbone can be changed in `./configs/train_classifier.yaml`

- Stage two: Meta-train HelixTransformer, run the following command line
```
# conv-4 backbone
python train_meta_helix_transformer.py
#resnet12 backbone
python train_meta_helix_transformer_resnet12.py
```
datasets/backbone/HelixTransformer model and other configs can be changes in `./configs/train_helix_transformer.yaml`
 The trained model can be tested by running the following command line:
 ```
 python test_helix_transformer.py
 ```
 datasets/model path and other configs can be changed in `./configs/test_helix_transformer.yaml`
 
You can also train/test our meta baseline(Relation Network) by running the following command line
```
python train_baseline.py
python test_baseline.py
```

## Selected few-shot classification results
Here we quote some performance comparisons from our paper on CUB, mini-ImageNet, tiered-ImageNet and mini-ImageNet &#8594; CUB.

<p align="center">
<img src="./imgs/cub_cropped.png" width="450">
</p>
<p align="center">
<img src="./imgs/cub_raw.png" width="450">
</p>
<p align="center">
<img src="./imgs/imagenet.png" width="750">
</p>
<p align="center">
<img src="./imgs/mini2cub.png" width="525">
</p>

## Contact
We have tried our best to verify the correctness of our released data, code and trained model weights. 
However, there are a large number of experiment settings, all of which have been extracted and reorganized from our original codebase. 
There may be some undetected bugs or errors in the current release. 
If you encounter any issues or have questions about using this code, please feel free to contact us via lt453@cornell.edu and dww78@cornell.edu.


# Autonomous Driving Perception



## Installation

```
cuda == 12.4
Python == 3.10.16
Pytorch == 2.6.0+cu124

# clone this repository
git clone https://github.com/NovelioPI/autonomous-driving-perception
```



## Datasets

We used Cityscapes dataset and CamVid dataset to train our model.  

- You can download cityscapes dataset from [here](https://www.cityscapes-dataset.com/). 

Note: please download leftImg8bit_trainvaltest.zip(11GB) and gtFine_trainvaltest(241MB). 

The Cityscapes dataset scripts for inspection, preparation, and evaluation can download from [here](https://github.com/mcordts/cityscapesScripts).


```
├── dataset  					# contains all datasets for the project
|  └── cityscapes 				#  cityscapes dataset
|  |  └── gtCoarse  		
|  |  └── gtFine 			
|  |  └── leftImg8bit 		
|  |  └── cityscapes_test_list.txt
|  |  └── cityscapes_train_list.txt
|  |  └── cityscapes_trainval_list.txt
|  |  └── cityscapes_val_list.txt
|  |  └── cityscapesscripts 	#  cityscapes dataset label convert scripts！
|  └── inform 	
|  |  └── camvid_inform.pkl
|  |  └── cityscapes_inform.pkl
|  └── camvid.py
|  └── cityscapes.py 

```



## Train

```
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 4.5e-2 --batch_size 4
```



## Test

```
python test.py --dataset cityscapes --checkpoint ./checkpoint/cityscapes/FBSNetbs4gpu1_train/model_1000.pth
```

## Predict
only for cityscapes dataset
```
python predict.py --dataset cityscapes 
```


## Acknowledgements

1. [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)
2. [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
3. [FBSNet: A Fast Bilateral Symmetrical Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/2109.00699v1)

# Global-Wheat-Detection: EfficientDet Inference

This is the inference part of my second solution to the [Global Wheat Detection](kaggle.com/c/global-wheat-detection) competition. This solution is based on a Pytorch implementation of EfficientDet-D5 and contains two notebooks: the [training notebook](https://github.com/jillerhaus/Global_Wheat_EffDet_Training) and this inference notebook.



## Prerequisites

This project requires the Jupyter notebook containing the completed project, one or more pre-trained EfficientDet-D5 weights (can be found in my [dataset](https://www.kaggle.com/johannesillerhaus/effdetweights), a CSV file containing information on ground-truth bounding box coordinates (labels.csv is included in this repository), as well as the [Global-Wheat-Detection dataset](https://www.kaggle.com/c/global-wheat-detection/data). labels.csv includes the ground truth boxes for the training dataset from the competition. The ground-truth coordinates are only necessary if you want to infer on a dataset that contains ground-truth information and want these boxes to be included in the plots of the predictions.

To run this project, you will need `Jupyter Notebooks` with a python >= 3.7 kernel to be installed on your machine. Instructions on how to install `Jupyter Notebooks` can be found on the [Jupyter website](https://jupyter.org/install).

The project itself is written in Python 3.7 and uses several different modules. To make it easier to use, a .yml file of the virtual anaconda environment I created for this solution will be included in this repository. Unfortunately some of the modules, such as NVIDIA Apex and the Pytorch implementation of EfficientDet itself need to be installed manually. Their repositories can be found in the attributions section of this readme. When using Windows, the pycocotools module will create issues. To remedy this, use `pip install pycocotools-windows`.

The file structure expected by the notebook is modeled after the Kaggle file structure, so that the notebook can be run on both Kaggle and Windows: The root directory should contain two directories, input and working. Place the extracted folder of the datasets (global-wheat-detection and effdet-weights) in input and the files contained in this repository in the working directories, respectively. This way the notebook will work without any changes to the code.

The notebook was used on Anaconda for Windows with both an NVIDIA 1050ti and a 2080ti and on Kaggle using an NVIDIA V100 and P100. The settings used in the notebook are for the 2080ti and the `batch_size` attribute in the `TrainGlobalConfig` class may need to be adjusted for the amount of VRAM of the GPU uses when infering.



## Attributions

The network used the Pytorch implementation of EfficientDet by rwightman found [here](github.com/rwightman/efficientdet-pytorch@75c10c855a0bd617f9b6be0835761121e924b999). 

As a starting point to develop my code I used the [\[Inference\] EfficientDet](https://www.kaggle.com/shonenkov/inference-efficientdet) notebook by Alex Shonenkov. 

The weighted box fusion code used was created by Roman Solovyev. A paper explaining the technique can be found [here](https://arxiv.org/abs/1910.13302)


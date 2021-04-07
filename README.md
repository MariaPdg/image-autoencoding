# image-autoencoding

In this task we consider Variational Autoencoder (VAE) in order to reconstruct and generate images for two datasets:
* [MS COCO](https://cocodataset.org/#download)
* [Food-101](https://www.kaggle.com/dansbecker/food-101)

## Project structure

* **configs:** all config files
* **data:** data loader
* **models:** vae architecture: encoder + decoder + re-parametrization layer

## Data setup

Specify data directories in ```configs/data_config.py```

Data should have the same data root: 
```python
# root where datasets are located (might be home directory)
data_root = 'datasets/'
```

### MS COCO

* [Training set](http://images.cocodataset.org/zips/train2017.zip) 

* [Testing set](http://images.cocodataset.org/zips/test2015.zip) 

* [Validation set](http://images.cocodataset.org/zips/test2014.zip)

Example directory: ```datasets/coco/[training_set]```

```python
# COCO dataset
dataset = 'coco'
coco_train_data = 'coco_train2017/train2017'
coco_valid_data = 'coco_valid2017/val2017'
coco_test_data = 'coco_test2017/test2017'
```

We use training and testing data for training. We allow to do this since VAE is trained in unsupervised manner. Evaluation is done with validation set.

### Food-101

* [Food-101](https://www.kaggle.com/dansbecker/food-101/download)

Example directory: ```datasets/food-101/[images]```

```python
# FOOD-101 dataset
dataset = 'food-101'
images_data = 'images'
meta_data = 'meta'
```


## Training

1. Set python path
```
export PYTHONPATH=$PYTHONPATH:[absolute path to the folder]/image-autoencoding/

```
2. Start training ```vae_train.py```

```
   python3 vae_train.py  -o [user path] -l [path to logs]
```
  Flags:
  * ``` -o [user_path]``` user path to save the results
  * ``` -l [path to logs]``` path to save logs
   
## Solution description

* We use VAE with the  following decoder and encoder: 
    
   <img src="docs/img.png" alt="arch" width="650"/>

* Image preprocessing is done in dataloader:
    - Image size = 64;
    - Center crop = 375;
    - Random Horizontal Flip;
    - Transform grey scale images to RGB image by channel replication;
    - Normalization (standardization) with mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5];
    
* Latent dimension = 128
  
* Loss function


## Results 

### MS COCO

#### Ground truth images

<img src="docs/img_gt.png" alt="ground truth" width="700"/>

#### **3 training epochs:**

<img src="docs/img_1.png" alt="3 epoch" width="700"/>


Reconstruction from sampled latent representations: 


<img src="docs/img_4.png" alt="3 epoch latent" width="700"/>


#### **20 training epochs:**


<img src="docs/img_2.png" alt="20 epoch" width="700"/>

Reconstruction from sampled latent representations: 

<img src="docs/img_3.png" alt="20 epoch latent" width="700"/>

Since we use natural scenes which are the most difficult for unsupervised training the images generated from the latent space 
are less meaningful in comparison to the models, which are trained with other more structural and uniform datasets, 
e.g. widely used in related works CelebA dataset with faces, LSUN bedrooms etc. 


### FOOD-101

Let's try tu use more uniform but smaller dataset FOOD-101. We also apply weight for KL divergence term.

#### Ground truth images

![img.png](docs/food_gt.png)

#### **3 training epochs:**

![img.png](docs/food1.png)


Reconstruction from sampled latent representations: 

![img.png](docs/food2.png)

#### **10 training epochs:**

![img.png](docs/food3.png)

Reconstruction from sampled latent representations: 

![img.png](docs/food4.png)


#### **20 training epochs:**

![img.png](docs/food5.png)

Reconstruction from sampled latent representations: 

![img.png](docs/food6.png)

Now the generated images are more similar to food images. Weighted Kl divergence leads to better reconstructions 
but the generated images  provide less information. Therefore, we have to achieve trade-off between the 
reconstruction error and VAE penalizer, which pushes the encoder distribution to be similar to the prior latent distribution. 

The VAE reconstructions are quite noisy. The reason is the lower dimension of the latent space comparing to the input images. 
Another reason is the sampling in the latent space. As a result, VAE penalizer pushes its reconstruction to the mean values of the latent representation
instead of real values. 

# image-autoencoding

In this task we consider Variational Autoencoder (VAE) in order to reconstruct and generate images from [MS COCO dataset](https://cocodataset.org/#download).

## Project structure

* **configs:** all config files
* **data:** data loader
* **models:** vae architecture: encoder + decoder + re-parametrization layer

## Data setup

Specify data directories in ```configs/data_config.py```

* [Training set](http://images.cocodataset.org/zips/train2017.zip) 

* [Testing set](http://images.cocodataset.org/zips/test2015.zip) 

* [Validation set](http://images.cocodataset.org/zips/test2014.zip)

We use training and testing data for training. We allow to do this since VAE is trained in unsupervised manner. Evaluation is done with validation set.

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
  * ``` -o [user_path]``` user path where to save the results
  * ``` -l [path to logs]``` path where to save logs
   
## Solution description

* We use VAE with the  following decoder and encoder: 

    ![img.png](docs/img.png)

* Image preprocessing is done in dataloader:
    - Image size = 64;
    - Center crop = 375;
    - Random Horizontal Flip;
    - Transform grey scale images to RGB image by channel replication;
    - Normalization (standardization) with mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5];
    
* Latent dimension = 128


## Results 

Ground truth images

![img](docs/img_gt.png)

#### **3 training epochs:**

![img_8.png](docs/img_1.png)


Reconstruction from the sampled latent representation: 

![img_11.png](docs/img_4.png)


#### **20 training epochs:**

![img_9.png](docs/img_2.png)


Reconstruction from the sampled latent representation: 

![img_10.png](docs/img_3.png)


Since we use natural scenes which are the most difficult for unsupervised training the images generated from the latent space 
are less meaningful in comparison to the models, which are trained with other more structural and uniform datasets, 
e.g. widely used in related works CelebA dataset with faces, LSUN bedrooms etc. 

The VAE reconstructions are quite noisy. The reason is the lower dimension of the latent space comparing to the input images. 
Another reason is the sampling in the latent space. As a result, VAE penalizer pushes its reconstruction to the mean values of the latent representation
instead of real values. 
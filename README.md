# image-autoencoding

In this task we consider Variational Autoencoder (VAE) in order to reconstruct and generate images for two datasets:
* [MS COCO](https://cocodataset.org/#download)
* [Food-101](https://www.kaggle.com/dansbecker/food-101)

## Project structure

* **configs:** all config files
* **data:** data loader
* **models:** vae architecture: encoder + decoder + re-parametrization layer

## Configs

* ```data_config.py:```   paths to datasets
* ```models_config.py:``` parameters for models
* ```train_config.py:```  training parameters

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

Example directory: ```datasets/coco/coco_train2017/train2017/```

```python
# COCO dataset
dataset = 'coco'
coco_train_data = 'coco_train2017/train2017'
coco_valid_data = 'coco_valid2017/val2017'
coco_test_data = 'coco_test2017/test2017'
```

We use training and testing data for training. We allow to do this since VAE is trained in unsupervised manner. Evaluation is done with the validation set.

### Food-101

* [Food-101](https://www.kaggle.com/dansbecker/food-101/download)

Example directory: ```datasets/food-101/images```

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
2. Specify training parameters in ```train_config.py```, e.g.

  ```python
  batch_size = 64
  learning_rate = 0.001
  weight_decay = 1e-7
  n_epochs = 100
  ```

2. Start training ```vae_train.py```

```
python3 vae_train.py  -o [user path] -l [path to logs]
```
  Flags:
  * ``` -o [user_path]``` user path to save the results
  * ``` -l [path to logs]``` path to save logs
   
## Solution description

* VAE is a widely used generative model, which is used in image reconstruction and generation tasks.
  It provides more efficient way (e.g. in comparison too PCA) to solve the dimensionality reduction problem for high dimensional data (e.g. text, images).
  A general VAE architecture is illustrated in Figure:
  
<p align="center">
<img src="docs/vae.png" alt="vae" width="550"/>
</p>

* A disadvantage of a simple autoencoder is its discrete latent space.
  As a result, there are some points which the autoencoder can not reconstruct. VAE provides a continuous latent space 
  due to KL divergence, which matches a prior distribution and a predicted encoder distribution (an approximate posterio). 

* We use VAE with the  following decoder and encoder: 
    
   <img src="docs/img.png" alt="arch" width="650"/>

* Image preprocessing is done in dataloader:
    - Image size = 64;
    - Center crop = 375;
    - Random Horizontal Flip;
    - Transform grey scale images to RGB image by channel replication;
    - Normalization (standardization) with mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5] 
      since we use Tanh aoutput activation function in the decoder.
     
    
* Latent dimension = 128
* Batch size = 64 (512 for the latent space visualization)
  
### Loss function

- VAE loss function represents the sum of the reconstruction error and KL divergence.

- Generally the reconstruction error corresponds to the mean square error between a ground thruth image and a reconstructed image 
  (here it is just a square error):
  
```python
# reconstruction error
mse = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2
```
- KL divergence is calculated for the Gaussian encoder distribution:
```
# kl divergence
kld = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
```

- Since we use separate optimizers for the encoder and the decoder networks we calculate the following loss function:

```python
if args.mode == 'vae':
        loss_encoder = (1 / batch_size) * torch.sum(kld)
        # loss_encoder = torch.sum(kld) 
        loss_decoder = torch.sum(recon_mse)
```
- Encoder loss corresponds to the penalizer term, which pushed the approximate posterior to the prior.
- Decoder loss corresponds to the reconstruction error. 
- We use KL-divergence weighted with batch_size or not in order to achieve a trade-off between two terms.
- We minimize the sum instead of mean values.

### Metrics

There is a huge issue regarding evaluation of generative models. We evaluated the reconstruction ability with 
the common image similarity metrics:

* Pearson corelation Coefficient (PCC)
* Structural Similarity (SSIM)

However, they are not capable to capture human perception.

## Results 

### MS COCO

#### Ground truth images

<img src="docs/img_gt.png" alt="ground truth" width="550"/>

#### **3 training epochs:**

<img src="docs/img_1.png" alt="3 epoch" width="550"/>


Reconstruction from sampled latent representations: 


<img src="docs/img_4.png" alt="3 epoch latent" width="550"/>


#### **20 training epochs:**


<img src="docs/img_2.png" alt="20 epoch" width="550"/>

Reconstruction from sampled latent representations: 

<img src="docs/img_3.png" alt="20 epoch latent" width="550"/>

Since we use natural scenes, which are the most difficult for unsupervised training, the images generated from the latent space 
are less meaningful in comparison to the models trained with other more structural and uniform datasets, 
e.g. widely used in related works CelebA dataset with faces, LSUN bedrooms etc. 


### FOOD-101

Let's try more uniform but smaller dataset FOOD-101. We also apply weight for KL divergence term.

#### Ground truth images

<img src="docs/food_gt.png" alt="food_gt" width="550"/>

#### **3 training epochs:**

<img src="docs/food1.png" alt="1" width="700"/>

Reconstruction from sampled latent representations: 

<img src="docs/food2.png" alt="2" width="700"/>

Latent space for the test set (alpha=1):


<img src="docs/lat1.png" alt="lat1" width="500"/>


#### **10 training epochs:**

<img src="docs/food3.png" alt="3" width="700"/>


Reconstruction from sampled latent representations: 

<img src="docs/food4.png" alt="4" width="700"/>


Latent space for the test set (alpha=1):

<img src="docs/lat2.png" alt="lat2" width="500"/>


#### **20 training epochs:**

<img src="docs/food5.png" alt="5" width="700"/>

Reconstruction from sampled latent representations: 

<img src="docs/food6.png" alt="6" width="700"/>

Latent space for the test set (alpha=1):

<img src="docs/lat3.png" alt="lat3" width="500"/>

(It seems longer training is required for clusterization in the latent space.) 

Now the generated images (from random latent vector) are more similar to food images. The weighted KL divergence leads to better reconstructions 
but the generated images  provide less information. 

## Conclusion

In order to achieve better reconstructions the latent variables must stay away from each other. Otherwise, 
they may coincide, as a consequence deteriorate the reconstructions. Therefore, we have to achieve trade-off between the 
reconstruction error and VAE penalizer, which pushes the encoder distribution to be similar to the prior latent distribution. 

The VAE reconstructions are quite noisy. The reason is the lower dimension of the latent space comparing to the input images. 
Another reason is the sampling in the latent space. As a result, VAE penalizer pushes its reconstruction to the mean values of the latent representation
instead of the real values. 

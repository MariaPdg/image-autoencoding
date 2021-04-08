"""____________Config file used for training____________"""

pretrained_gan = None  # None or pretrained model, e.g. 'vae_20210203-173210'
load_epoch = 395       # number of loaded epoch
evaluate = False       # if you want evaluate only

# Image parameters
image_crop = 375
image_size = 64
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# latent space dimension
latent_dim = 128

# Device for training: cpu or cuda
device = 'cuda:5'

# Other training parameters
patience = 0   # for early stopping, 0 = deactivate early stopping
batch_size = 512
learning_rate = 0.001
weight_decay = 1e-7
n_epochs = 100
num_workers = 4
step_size = 20  # for scheduler
gamma = 0.1     # for scheduler
lambda_mse = 1e-6
decay_lr = 0.75
decay_mse = 1
beta = 1.0





pretrained_gan = None  # e.g. 'vae_20210203-173210'
load_epoch = 395
evaluate = False

patience = 0   # for early stopping, 0 = deactivate early stopping

image_crop = 375
image_size = 64
latent_dim = 128

data_split = 0.2
batch_size = 512
learning_rate = 0.001
weight_decay = 1e-7
n_epochs = 100
num_workers = 4
step_size = 20  # for scheduler
gamma = 0.1     # for scheduler
lambda_mse = 1e-6
decay_lr = 0.75
decay_margin = 1
decay_mse = 1
beta = 1.0

device = 'cuda:5'

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


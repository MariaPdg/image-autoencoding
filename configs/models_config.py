"""______Config file used for model definition_________"""

# parameters for conv layer in encoder and decoder
kernel_size = 5
stride = 2
padding = 2
dropout = 0.7

# channels for conv layers
encoder_channels = [64, 128, 256]
decoder_channels = [256, 128, 32, 3]
discrim_channels = [32, 128, 256, 256, 512]

# settings for resolution 64
image_size = 64
fc_input = 8
fc_output = 1024
fc_input_gan = 8
fc_output_gan = 512
stride_gan = 1
latent_dim = 128
output_pad_dec = [True, True, True]

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import configs.models_config as config


class EncoderBlock(nn.Module):

    """CNN-based encoder block"""

    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=config.kernel_size,
                              padding=config.padding, stride=config.stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):

        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)

        return ten


class DecoderBlock(nn.Module):

    """CNN-based decoder block"""

    def __init__(self, channel_in, channel_out, out=False):
        super(DecoderBlock, self).__init__()

        # Settings for settings from different papers
        if out:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size, padding=config.padding,
                                           stride=config.stride, output_padding=1,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size,
                                           padding=config.padding,
                                           stride=config.stride,
                                           bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):

        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)

        return ten


class Encoder(nn.Module):

    """ VAE-based encoder"""

    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()

        self.size = channel_in
        layers_list = []
        for i in range(3):
            layers_list.append(EncoderBlock(channel_in=self.size, channel_out=config.encoder_channels[i]))
            self.size = config.encoder_channels[i]
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=config.fc_input * config.fc_input * self.size,
                                          out_features=config.fc_output, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_output, momentum=0.9),
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=config.fc_output, out_features=z_size)
        self.l_var = nn.Linear(in_features=config.fc_output, out_features=z_size)

    def forward(self, ten):

        """
        :param ten: input image
        :return: mu: mean value
        :return: logvar: log of variance for numerical stability
        """

        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)

        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):

    """ VAE-based decoder"""

    def __init__(self, z_size, size):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=config.fc_input * config.fc_input * size, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_input * config.fc_input * size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size, out=config.output_pad_dec[0]))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[1], out=config.output_pad_dec[1]))
        self.size = config.decoder_channels[1]
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[2], out=config.output_pad_dec[2]))
        self.size = config.decoder_channels[2]
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=config.decoder_channels[3], kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        """
        :param ten: re-parametrized latent variable
        :return: reconstructed image
        """
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, config.fc_input, config.fc_input)
        ten = self.conv(ten)

        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class VAE(nn.Module):

    """VAE model: encoder + decoder + re-parametrization layer"""

    def __init__(self, device, z_size=128):
        super(VAE, self).__init__()

        self.z_size = z_size  # latent space size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.init_parameters()
        self.device = device

    def init_parameters(self):

        """Glorot initialization"""

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def reparametrize(self, mu, logvar):

        """ Re-parametrization trick"""

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())

        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparametrize(mus, log_variances)
            x_tilde = self.decoder(z)

            # generate from random latent variable
            z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)
            return x_tilde, x_p, mus, log_variances, z_p

        else:
            if x is None:
                z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p

            else:
                mus, log_variances = self.encoder(x)
                z = self.reparametrize(mus, log_variances)
                x_tilde = self.decoder(z)
                return x_tilde

    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(x, x_tilde, mus, variances):

        """
        VAE loss: reconstruction error + KL divergence

        :param x: ground truth image
        :param x_tilde: reconstruction from the decoder
        :param mus: mean value from the encoder
        :param variances: log var from the encoder
        :return: mse: reconstruction error
        :return: kld: kl divergence
        """

        # reconstruction error
        mse = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl divergence
        kld = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        return mse, kld


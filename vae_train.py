import os
import time
import numpy
import json
import torch
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

import configs.train_config as gan_cfg
import configs.data_config as data_cfg
from models.vae import VAE
from train_utils import evaluate, PearsonCorrelation, StructuralSimilarity
from data.data_loader import CocoDataloader, GreyToColor, Food101Dataloader

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

DEBUG = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='user path where to save', type=str)
    parser.add_argument('--logs', '-l', help='path where to save logs', type=str)
    parser.add_argument('--batch_size', '-b', default=gan_cfg.batch_size, help='batch size for dataloader', type=int)
    parser.add_argument('--learning_rate', '-lr', default=gan_cfg.learning_rate, help='learning rate', type=float)
    parser.add_argument('--epochs', '-e', default=gan_cfg.n_epochs, help='number of epochs', type=int)
    parser.add_argument('--image_crop', '-im_crop', default=gan_cfg.image_crop, help='size to which image should '
                                                                                     'be cropped', type=int)
    parser.add_argument('--image_size', '-im_size', default=gan_cfg.image_size, help='size to which image should '
                                                                                     'be scaled', type=int)
    parser.add_argument('--device', '-d', default=gan_cfg.device, help='what device to use', type=str)
    parser.add_argument('--num_workers', '-nw', default=gan_cfg.num_workers, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--step_size', '-step', default=gan_cfg.step_size, help='number of epochs after which '
                                                                                'to decrease learning rate', type=int)
    parser.add_argument('--patience', '-p', default=gan_cfg.patience, help='number of epochs with unchanged lr '
                                                                           'for early stopping', type=int)
    parser.add_argument('--weight_decay', '--wd', default=gan_cfg.weight_decay, help='weight decay used by optimizer',
                        type=float)
    parser.add_argument('--latent_dim', '-lat_dim', default=gan_cfg.latent_dim, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--message', '-m', default='default message', help='experiment description', type=str)
    parser.add_argument('--pretrained_gan', '-pretrain', default=gan_cfg.pretrained_gan, help='pretrained gan', type=str)
    parser.add_argument('-load_epoch', '-pretrain_epoch', default=gan_cfg.load_epoch, help='epoch of the pretrained model',
                        type=int)
    parser.add_argument('--lambda_mse', default=gan_cfg.lambda_mse, type=float, help='weight for style error')
    parser.add_argument('--decay_mse', default=gan_cfg.decay_mse, type=float, help='mse weight decrease')
    parser.add_argument('--decay_lr', default=gan_cfg.decay_lr, type=float, help='learning rate decay for lr scheduler')
    parser.add_argument('--beta', default=gan_cfg.beta, type=float, help='beta factor for beta-vee')
    parser.add_argument('--mode', default='vae', help='vae, beta-vae')
    parser.add_argument('--dataset', default=data_cfg.dataset, help='coco, food-101')

    args = parser.parse_args()

    # Path to pickle file with bold5000 data
    USER_ROOT = args.output
    SAVE_PATH = os.path.join(USER_ROOT, data_cfg.save_training_results)

    # Create directory to save weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'train_gan_' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info("Used device: %s" % device)

    logging.info('set up random seeds')
    torch.manual_seed(12345)

    # Create directory for results
    if DEBUG:
        saving_dir = os.path.join(SAVE_PATH, 'debug', 'debug_gan_{}'.format(timestep))
    else:
        saving_dir = os.path.join(SAVE_PATH, 'gan', 'gan_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    if args.pretrained_gan is not None:
        pretrained_model_dir = os.path.join(SAVE_PATH, 'gan', args.pretrained_gan, args.pretrained_gan + '.pth')
    saving_name = os.path.join(saving_dir, 'gan_{}.pth'.format(timestep))

    # Save arguments
    with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset == 'food-101':

        FOOD_IMAGES = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.images_data)
        FOOD_TRAIN_META = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.meta_data, 'train.txt')
        FOOD_TEST_META = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.meta_data, 'test.txt')

        with open(FOOD_TRAIN_META) as f:
            train_image_names = f.readlines()
        with open(FOOD_TEST_META) as f:
            test_image_names = f.readlines()

        # Load data
        training_data = Food101Dataloader(FOOD_IMAGES, train_image_names,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(gan_cfg.mean,
                                                                                            gan_cfg.std)
                                                                       ]))

        # Load data
        validation_data = Food101Dataloader(FOOD_IMAGES, test_image_names,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(gan_cfg.mean,
                                                                                            gan_cfg.std)
                                                                       ]))

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'coco':

        COCO_TEST_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.coco_test_data)
        COCO_TRAIN_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.coco_train_data)
        COCO_VALID_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.dataset, data_cfg.coco_valid_data)

        # Load data
        training_data = CocoDataloader(COCO_TRAIN_DATA,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(gan_cfg.mean,
                                                                                            gan_cfg.std)
                                                                       ]))
        validation_data = CocoDataloader(COCO_VALID_DATA,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(gan_cfg.mean,
                                                                                            gan_cfg.std)
                                                                       ]))
        test_data = CocoDataloader(COCO_TEST_DATA,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(gan_cfg.mean,
                                                                                            gan_cfg.std)
                                                                       ]))
        train_test_data = ConcatDataset([training_data, test_data])

        dataloader_train = DataLoader(train_test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    else:
        logging.info('Specify dataset')

    writer = SummaryWriter(saving_dir + '/runs_' + timestep)
    writer_encoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/encoder')
    writer_decoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/decoder')

    model = VAE(device=device, z_size=args.latent_dim).to(device)

    if args.pretrained_gan is not None and os.path.exists(pretrained_model_dir.replace(".pth", ".csv")):

        logging.info('Load pretrained model')
        model_dir = pretrained_model_dir.replace(".pth", '_{}.pth'.format(args.load_epoch))
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        results = pd.read_csv(pretrained_model_dir.replace(".pth", ".csv"))
        results = {col_name: list(results[col_name].values) for col_name in results.columns}
        stp = 1 + len(results['epochs'])
        if gan_cfg.evaluate:
            images_dir = os.path.join(saving_dir, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            pcc, ssim, mse, is_mean = evaluate(model, dataloader_valid, norm=True, mean=gan_cfg.mean, std=gan_cfg.std,
                                               path=images_dir)
            print("Mean PCC:", pcc)
            print("Mean SSIM:", ssim)
            print("Mean MSE:", mse)
            print("IS mean", is_mean)
            exit(0)
    else:
        logging.info('Initialize')
        stp = 1

    results = dict(
        epochs=[],
        loss_encoder=[],
        loss_decoder=[]
    )

    # An optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=args.learning_rate, alpha=0.9,
                                            eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)
    optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=args.learning_rate, alpha=0.9,
                                            eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)

    # Metrics
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity(mean=gan_cfg.mean, std=gan_cfg.std)
    mse_loss = nn.MSELoss()

    result_metrics_train = {}
    result_metrics_valid = {}
    metrics_train = {'train_PCC': pearson_correlation, 'train_SSIM': structural_similarity, 'train_MSE': mse_loss}
    metrics_valid = {'valid_PCC': pearson_correlation, 'valid_SSIM': structural_similarity, 'valid_MSE': mse_loss}

    if metrics_valid is not None:
        for key in metrics_valid.keys():
            results.update({key: []})
        for key, value in metrics_valid.items():
            result_metrics_valid.update({key: 0.0})

    if metrics_train is not None:
        for key in metrics_train.keys():
            results.update({key: []})
        for key, value in metrics_train.items():
            result_metrics_train.update({key: 0.0})

    batch_number = len(dataloader_train)
    step_index = 0

    for idx_epoch in range(args.epochs):

        for batch_idx, data_batch in enumerate(dataloader_train):

            model.train()

            if args.dataset == 'coco':
                batch_size = len(data_batch)

                x = Variable(data_batch, requires_grad=False).float().to(device)

            elif args.dataset == 'food-101':
                batch_size = len(data_batch['image'])

                x = Variable(data_batch['image'], requires_grad=False).float().to(device)
                labels = data_batch['label']

            # Take model predictions/reconstruction
            x_tilde, x_p, mus, log_variances, z_p = model(x)

            # Calculate reconstruction error and KL divergence
            recon_mse, kld = VAE.loss(x, x_tilde, mus, log_variances)

            # Train encoder and decoder
            train_enc = True
            train_dec = True

            # Beta-VAE/GAN loss
            if args.mode == 'beta-vae':
                beta = args.beta
                kld_weight = 1 / batch_size
                loss_encoder = torch.sum(kld) * beta * kld_weight + torch.sum(recon_mse)
                loss_decoder = torch.sum(args.lambda_mse * recon_mse)

            # VAE loss
            if args.mode == 'vae':
                loss_encoder = (1 / batch_size) * torch.sum(kld) + torch.sum(recon_mse)
                # loss_encoder = torch.sum(kld) + torch.sum(recon_mse)
                loss_decoder = torch.sum(args.lambda_mse * recon_mse)

            # Register mean values
            loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
            loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size

            # Backpropagation
            # clean grads
            model.zero_grad()

            if train_enc:
                # encoder
                loss_encoder.backward(retain_graph=True)
                # we can clamp the grad here
                # [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                # update parameters
                optimizer_encoder.step()
                # clean others, so they are not afflicted by encoder loss
                model.zero_grad()

            # Decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                # [p.grad.data.clamp_(-1, 1) for p in model.decoder.parameters()]
                optimizer_decoder.step()

            logging.info(
                f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                f'---- encoder loss: {loss_encoder_mean:.5f} ---- | '
                f'---- decoder loss: {loss_decoder_mean:.5f} ---- | ')

            writer_encoder.add_scalar('loss_encoder_batch', loss_encoder_mean, step_index)
            writer_decoder.add_scalar('loss_decoder_batch', loss_decoder_mean, step_index)

            step_index += 1

        # End of epoch
        lr_encoder.step()
        lr_decoder.step()

        writer_encoder.add_scalar('loss_encoder', loss_encoder_mean, idx_epoch)
        writer_decoder.add_scalar('loss_decoder', loss_decoder_mean, idx_epoch)

        if not idx_epoch % 2:
            # Save train examples
            images_dir = os.path.join(saving_dir, 'images', 'train')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            # Ground truth
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(x[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
            plt.savefig(gt_dir)

            # Reconstructed images
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(x_tilde[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
            plt.savefig(output_dir)

            # Visualize latent space for training set
            plt.figure(figsize=(10, 6))
            plt.scatter(mus.cpu().detach()[..., 0], mus.cpu().detach()[..., 1], c=labels, cmap='rainbow')
            plt.colorbar()
            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_latent')
            plt.savefig(output_dir)
            plt.show()

        logging.info('Evaluation')

        for batch_idx, data_batch in enumerate(dataloader_valid):

            model.eval()

            if args.dataset == 'coco':
                batch_size = len(data_batch)

                data_in = Variable(data_batch, requires_grad=False).float().to(device)
                data_target = Variable(data_batch, requires_grad=False).float().to(device)

            elif args.dataset == 'food-101':
                batch_size = len(data_batch['image'])

                data_in = Variable(data_batch['image'], requires_grad=False).float().to(device)
                data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                labels = data_batch['label']

            out = model(data_in)

            # Validation metrics for the first validation batch
            if metrics_valid is not None:
                for key, metric in metrics_valid.items():
                    if key == 'cosine_similarity':
                        result_metrics_valid[key] = metric(out, data_target).mean()
                    else:
                        result_metrics_valid[key] = metric(out, data_target)

            # Training metrics for the last training batch
            if metrics_train is not None:
                for key, metric in metrics_train.items():
                    if key == 'cosine_similarity':
                        result_metrics_train[key] = metric(x_tilde, x).mean()
                    else:
                        result_metrics_train[key] = metric(x_tilde, x)

            # Save validation examples
            images_dir = os.path.join(saving_dir, 'images', 'valid')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
                os.makedirs(os.path.join(images_dir, 'random'))

            out = out.data.cpu()

            # Visualize latent space for validation set
            mus, var = model.encoder(data_in)
            plt.figure(figsize=(10, 6))
            plt.scatter(mus.cpu().detach()[..., 0], mus.cpu().detach()[..., 1], c=labels, cmap='rainbow')
            plt.colorbar()
            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_latent')
            plt.savefig(output_dir)
            plt.show()

            if idx_epoch == 0:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(data_in[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                plt.savefig(gt_dir)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
            plt.savefig(output_dir)

            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            writer.add_image("reconstructed", out, step_index)

            out = model(None, 25)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=5)
            writer.add_image("generated", out, step_index)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            output_dir = os.path.join(images_dir, 'random', 'epoch_' + str(idx_epoch) + '_output_' + 'rand')
            plt.savefig(output_dir)

            out = data_target.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            writer.add_image("original", out, step_index)

            if metrics_valid is not None:
                for key, values in result_metrics_valid.items():
                    result_metrics_valid[key] = torch.mean(values)
                # Logging metrics
                if writer is not None:
                    for key, values in result_metrics_valid.items():
                        writer.add_scalar(key, values, stp + idx_epoch)

            if metrics_train is not None:
                for key, values in result_metrics_train.items():
                    result_metrics_train[key] = torch.mean(values)
                # Logging metrics
                if writer is not None:
                    for key, values in result_metrics_train.items():
                        writer.add_scalar(key, values, stp + idx_epoch)

            logging.info(
                f'Epoch  {idx_epoch} ---- train PCC:  {result_metrics_train["train_PCC"].item():.5f} ---- | '
                f'---- train SSIM: {result_metrics_train["train_SSIM"].item():.5f} ---- '
                f'---- train MSE: {result_metrics_train["train_MSE"].item():.5f} ---- ')

            logging.info(
                f'Epoch  {idx_epoch} ---- valid PCC:  {result_metrics_valid["valid_PCC"].item():.5f} ---- | '
                f'---- valid SSIM: {result_metrics_valid["valid_SSIM"].item():.5f} ---- '
                f'---- valid MSE: {result_metrics_valid["valid_MSE"].item():.5f} ---- ')

            # only for one batch
            break

        if not idx_epoch % 10:
            torch.save(model.state_dict(), saving_name.replace('.pth', '_' + str(idx_epoch) + '.pth'))
            logging.info('Saving model')

            # Record losses & scores
        results['epochs'].append(idx_epoch + stp)
        results['loss_encoder'].append(loss_encoder_mean)
        results['loss_decoder'].append(loss_decoder_mean)

        if metrics_valid is not None:
            for key, value in result_metrics_valid.items():
                metric_value = torch.tensor(value, dtype=torch.float64).item()
                results[key].append(metric_value)

        if metrics_train is not None:
            for key, value in result_metrics_train.items():
                metric_value = torch.tensor(value, dtype=torch.float64).item()
                results[key].append(metric_value)

        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(saving_name.replace(".pth", ".csv"), index=False)


    exit(0)

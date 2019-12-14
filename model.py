import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.models.inception import inception_v3
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import sys
import numpy as np
import argparse
import os
import math
from scipy.stats import entropy
import kornia.filters as filters

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_pred(inception_model,x):
    x = inception_model(x)
    return F.softmax(x).data.cpu().numpy()


def inception_score(dataloader, batch_size, N, resize=True, splits=3):

    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    
    # Set up dataloader
    inception_model = inception_v3(pretrained=True, transform_input=False).type(Tensor)
    inception_model.eval();


    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch) in enumerate(dataloader):
        if resize:
            batch =  F.interpolate(batch,size=(299, 299), mode='bilinear', align_corners=True).type(Tensor)

        preds[i*batch_size:(i+1)*batch_size] = get_pred(inception_model,batch)
        del batch
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--dataset", type=str, default="cifar", help="dataset cifar-10 or mnist")
parser.add_argument("--blur", type=str, default=None, help="box blur or interpolate blur of real images")
parser.add_argument("--folder_name", type=str, default="sample", help="folder name for save generated images")
parser.add_argument("--img_as_input", type=bool, default=False, help="take images as input")
parser.add_argument("--restrict_learning", type=str, default=None, help="restricting discriminator loss by threshold or differential")

opt = parser.parse_args()
print(opt)

cuda = True

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
# Configure data loader

if opt.dataset is "mnist":
    os.makedirs("images_mnist", exist_ok=True)
    os.makedirs("./data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last= True
    )

if opt.dataset is "cifar":
    os.makedirs("images_cifar_%s" % opt.folder_name, exist_ok=True)
    os.makedirs("./data/cifar-10", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data/cifar",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last= True
    )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        if opt.blur=="box":
            p=max(1,int(8*(1-epoch/opt.n_epochs)))
            filter_imgs=filters.box_blur(imgs,kernel_size=(p,p))
        elif opt.blur=="interpolate":
            p=(epoch*len(dataloader)+i)/(opt.n_epochs*len(dataloader))
            p=min(1,p+0.1)
            filter_imgs = F.interpolate(imgs, size=None, scale_factor=p, mode='bilinear')
            filter_imgs = F.interpolate(imgs, size=[opt.img_size, opt.img_size], scale_factor=None, mode='bilinear')
        else:
            filter_imgs = imgs

        real_imgs = Variable(imgs.type(Tensor))
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()


        # Sample noise as generator input
        

        if opt.img_as_input is True:
            if epoch==0 and i==0:
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            else:
                resized=F.max_pool2d(gen_imgs,2,stride=2)
                resized=resized.sum(dim=1)
                resized=resized.reshape(imgs.shape[0],opt.latent_dim)
                resized=F.normalize(resized)
                p=(epoch*len(dataloader)+i)/(opt.n_epochs*len(dataloader))
                p=4*p*(1-p)
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))*(1-p) + resized*p)
        
        else:
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))


        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()


        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2


        if "restrict_learning" == "threshold":
            if (epoch == 0 and i < 100) or d_loss.item() >0.6:
                d_loss.backward()
                optimizer_D.step()
        elif "restrict_learning" == "differential":
            if (epoch == 0 and i < 100) or math.abs(d_loss.item()-g_loss.item() < 0.1):
                d_loss.backward()
                optimizer_D.step()
        else:
            d_loss.backward()
            optimizer_D.step()




        if i==0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
        if i==0:
            batch_size=3
            d = []
            for j, (b) in enumerate(gen_imgs):
                if opt.channels==3:
                    a = b
                else:
                    a = torch.cat([b, b, b], dim=0)
                if j % batch_size == 0:
                    c = a
                else:
                    c = torch.cat([c, a], dim=0)
                    if j % batch_size == batch_size - 1:
                        d.append(c.view(batch_size,3,32,32))
            print(inception_score(d, batch_size = batch_size, N = len(gen_imgs)))

        batches_done = epoch * len(dataloader) + i

        if batches_done % 1000 == 0:
            save_image(gen_imgs.data[:25], "images_%s_%s/%d.png" % (opt.dataset, opt.folder_name, batches_done), nrow=5, normalize=True)

for i in range(opt.batch_size):
    os.makedirs("images_cifar_%s/result" % opt.folder_name, exist_ok=True)
    save_image(gen_imgs.data[i], "images_%s_%s/result/%d.png" % (opt.dataset, opt.folder_name, i))
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import config
from typing import Any
from config import nc, ndf

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu 
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config.nz, out_channels=config.num_generator_features * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=config.num_generator_features * 8),
            nn.ReLU(True),
            # Size of output is kernel_size^2 * num_generator_features * 8 = 4x4x(8x64)

            nn.ConvTranspose2d(in_channels=config.num_generator_features*8, out_channels=config.num_generator_features*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.num_generator_features*4),
            nn.ReLU(True),
            # Size of output is kernel_size^2 * num_generator_features * 4 = 4x4x(4x64)

            nn.ConvTranspose2d(in_channels=config.num_generator_features*4, out_channels=config.num_generator_features*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.num_generator_features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=config.num_generator_features*2, out_channels=config.num_generator_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=config.num_generator_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=config.num_generator_features, out_channels=config.IMAGE_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()            
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=0.0, std=0.02)
    elif classname.find("Batch") != -1:
        nn.init.normal_(tensor=m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(tensor=m.bias.data, val=0)


def setup() -> torch.device:
    device = torch.device("cuda:0" if config.CUDA else "cpu")
    cudnn.benchmark = True
    return device

def show_training_data(data: torch.utils.data.DataLoader, device: torch.device) -> None:
    real_batch = next(iter(data))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64]).cpu(),(1,2,0)))
    plt.show()



def main():
    device = setup()
    data = dset.MNIST(root = "model", download=True,
                    transform=transforms.Compose([
                    transforms.Resize(config.X_DIM),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ]))
    dataloader = torch.utils.data.DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True)

    #show_training_data(dataloader, device)

    nnGenerator = Generator(config.ngpu).to(device)
    nnDiscriminator = Discriminator(config.ngpu).to(device)

    nnGenerator.apply(weights_init)
    nnDiscriminator.apply(weights_init)

    # Print the model
    print(nnDiscriminator)
    print(nnGenerator)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, config.Z_DIM, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(nnDiscriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(nnGenerator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(config.EPOCH_NUM):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            nnDiscriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = nnDiscriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = nnGenerator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = nnDiscriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            nnGenerator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = nnDiscriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, config.EPOCH_NUM, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config.EPOCH_NUM-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = nnGenerator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if (epoch) % 2 == 0:  # Save every 2 epochs
            checkpoint = {
                "epoch": epoch,
                "generator_state_dict": nnGenerator.state_dict(),
                "discriminator_state_dict": nnDiscriminator.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
            }
            torch.save(checkpoint, f"dcgan_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

            
    torch.save(checkpoint, 'dcgan_final.pth')






if __name__ == "__main__":
    main()

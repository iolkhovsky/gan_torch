import torch
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
from tqdm import tqdm
from models.discriminator import FaceDiscriminator
from models.generator import FaceGenerator
from dataset.faces_dataset import make_dataloader
from torch.optim import Adam, SGD
from metrics import accuracy
from dataset.utils import decode_img, array_yxc2cyx
from dataset.faces_dataset import FacesDataset, DEFAULT_RGB_MEAN, DEFAULT_RGB_STD, AddGaussianNoise

from utils import get_readable_timestamp


def train(generator, discriminator, train_loader, optimizer_gen, optimizer_discr, epoch_id=0, scheduler=None, device="cpu",
          autosave_period=None, valid_period=None, tb_writer=None, transform=None):

    with tqdm(total=len(train_loader) * train_loader.batch_size,
              desc=f'Epoch {epoch_id + 1}',
              unit='image') as pbar:
        for batch_idx, real_imgs in enumerate(train_loader):

            generator.train()
            discriminator.train()

            optimizer_discr.zero_grad()
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            label = torch.full((b_size,), 1, device=device)
            output = discriminator(real_imgs)
            loss_function = BCELoss(reduction="mean")
            errD_real = loss_function(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            random_descriptors = torch.randn(b_size, 64, device=device)
            # Generate fake image batch with G
            fake = generator(random_descriptors)
            if transform is not None:
                fake = transform(fake)
            label.fill_(0)
            # Classify all fake batch with D
            output = discriminator(fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = loss_function(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_discr.step()

            optimizer_gen.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake)
            # Calculate G's loss based on this output
            errG = loss_function(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_gen.step()

            global_step = epoch_id * len(train_loader) + batch_idx
            if tb_writer:
                tb_writer.add_scalar("Loss/Discriminator", errD.item(), global_step)
                tb_writer.add_scalar("Loss/DiscriminatorFake", errD_fake.item(), global_step)
                tb_writer.add_scalar("Loss/DiscriminatorReal", errD_real.item(), global_step)
                tb_writer.add_scalar("Loss/Generator", errG.item(), global_step)
                tb_writer.add_scalar("Loss/Total", errG.item() + errD.item(), global_step)
                tb_writer.add_scalar("Accuracy/DiscrReal", D_x, global_step)
                tb_writer.add_scalar("Accuracy/DiscrFake", 1.0 - D_G_z1, global_step)
                tb_writer.add_scalar("Accuracy/GenFake", D_G_z2, global_step)

            if valid_period:
                if (batch_idx + 1) % valid_period == 0:
                    with torch.no_grad():

                        gen_imgs = []
                        for i, img in enumerate(fake):
                            if img.device != "cpu":
                                img = img.cpu()
                            img = decode_img(img.detach().numpy())
                            img = array_yxc2cyx(img)
                            gen_imgs.append(torch.from_numpy(img))

                        target_imgs = []
                        for i, img in enumerate(real_imgs[:min(len(real_imgs), 8)]):
                            if img.device != "cpu":
                                img = img.cpu()
                            img = decode_img(img.detach().numpy())
                            img = array_yxc2cyx(img)
                            target_imgs.append(torch.from_numpy(img))

                        if tb_writer:
                            img_grid_gen = torchvision.utils.make_grid(gen_imgs)
                            tb_writer.add_image('Valid/Generated', img_tensor=img_grid_gen,
                                                    global_step=global_step, dataformats='CHW')
                            img_grid_tgt = torchvision.utils.make_grid(target_imgs)
                            tb_writer.add_image('Valid/Real', img_tensor=img_grid_tgt,
                                                    global_step=global_step, dataformats='CHW')

            if autosave_period is not None:
                if (batch_idx + 1) % autosave_period == 0:
                    model_name = str(generator) + "_" + get_readable_timestamp() + "_epoch_" + \
                               str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                    torch.save(generator.state_dict(), model_name)
                    print(model_name, " has been saved")
                    model_name = str(discriminator) + "_" + get_readable_timestamp() + "_epoch_" + \
                               str(epoch_id) + "_batch_" + str(batch_idx) + ".pt"
                    torch.save(discriminator.state_dict(), model_name)
                    print(model_name, " has been saved")

            pbar.update(train_loader.batch_size)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch-train", type=int, default=32,
                        help="Size of batch for training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Target device: cpu/cuda")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Type of optmizer")
    parser.add_argument("--autosave-period", type=int, default=0,
                        help="Period of model autosave")
    parser.add_argument("--autosave-period-unit", type=str, default="e",
                        help="Units for autosave (e/b)")
    parser.add_argument("--valid-period", type=int, default=10,
                        help="Period of validation")
    parser.add_argument("--valid-period-unit", type=str, default="e",
                        help="Units for validation (e/b)")
    parser.add_argument("--pretrained_gen", type=str,
                        help="Abs path to pretrained generator")
    parser.add_argument("--pretrained_disc", type=str,
                        help="Abs path to pretrained discriminator")
    parser.add_argument("--scheduler", type=int, default=0,
                        help="Use lr scheduler or not")
    parser.add_argument("--l2", type=float, default=0,
                        help="L2 reularization coefficient")
    parser.add_argument("--noise-mean", type=float,
                        help="Gaussian noise mean")
    parser.add_argument("--noise-std", type=float,
                        help="Gaussian noise std")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    generator = FaceGenerator()
    generator = generator.to(args.device)
    if args.pretrained_gen:
        generator.load_state_dict(torch.load(args.pretrained_gen))

    discriminator = FaceDiscriminator()
    discriminator = discriminator.to(args.device)
    if args.pretrained_disc:
        discriminator.load_state_dict(torch.load(args.pretrained_disc))

    transform = None
    if (args.noise_mean is not None) and (args.noise_std is not None):
        transform = AddGaussianNoise(mean=args.noise_mean, std=args.noise_std)
    dataset = FacesDataset("/home/igor/datasets/faces6k/aligned", target_size=(64, 64),
                           mean=DEFAULT_RGB_MEAN, std=DEFAULT_RGB_STD, transform=transform)
    train_dloader = make_dataloader(dataset, batch_size=args.batch_train, shuffle_dataset=True)

    optimizer_gen = None
    if args.optimizer == "adam":
        optimizer_gen = Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999),
                         weight_decay=args.l2)
    elif args.optimizer == "sgd":
        optimizer_gen = SGD(generator.parameters(), lr=args.learning_rate,
                        weight_decay=args.l2)

    optimizer_discr = None
    if args.optimizer == "adam":
        optimizer_discr = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999),
                         weight_decay=args.l2)
    elif args.optimizer == "sgd":
        optimizer_discr = SGD(discriminator.parameters(), lr=args.learning_rate,
                        weight_decay=args.l2)

    scheduler = None

    tboard_writer = SummaryWriter()

    try:
        for e in range(args.epochs):
            train(generator, discriminator, train_dloader, optimizer_gen, optimizer_discr, e, scheduler=scheduler,
                  device=args.device, autosave_period=None, valid_period=args.valid_period, tb_writer=tboard_writer,
                  transform=transform)
        model_name = "pretrained_models/" + str(generator) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(generator.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
        model_name = "pretrained_models/" + str(discriminator) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(discriminator.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
    except KeyboardInterrupt:
        model_name = "pretrained_models/" + str(generator) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(generator.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
        model_name = "pretrained_models/" + str(discriminator) + "_completed_" + get_readable_timestamp() + ".pt"
        torch.save(discriminator.state_dict(), model_name)
        print("Training completed. Final model " + model_name + " has been saved")
    return


if __name__ == "__main__":
    main()

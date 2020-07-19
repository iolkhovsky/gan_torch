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
from dataset.faces_dataset import FacesDataset, DEFAULT_RGB_MEAN, DEFAULT_RGB_STD

from utils import get_readable_timestamp


def train(generator, discriminator, train_loader, optimizer, epoch_id=0, scheduler=None, device="cpu",
          autosave_period=None, valid_period=None, tb_writer=None):

    generator.train()
    discriminator.train()

    with tqdm(total=len(train_loader) * train_loader.batch_size,
              desc=f'Epoch {epoch_id + 1}',
              unit='image') as pbar:
        for batch_idx, real_imgs in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            random_descriptors = torch.rand(len(real_imgs), 64).to(device)
            artif_imgs = generator.forward(random_descriptors)
            inputs = torch.cat((real_imgs, artif_imgs), 0)
            labels = torch.Tensor(len(real_imgs) * [1] + len(real_imgs) * [0]).to(device)
            one_hot = torch.Tensor(len(real_imgs) * [[0, 1]] + len(real_imgs) * [[1, 0]]).to(device)
            optimizer.zero_grad()
            output = discriminator(inputs)
            loss_function = BCELoss(reduction="mean")
            train_loss = loss_function(output, one_hot)
            train_loss.backward()
            optimizer.step()

            global_step = epoch_id * len(train_loader) + batch_idx
            if tb_writer:
                tb_writer.add_scalar("Loss/Train", train_loss.item(), global_step)

            if valid_period:
                if (batch_idx + 1) % valid_period == 0:
                    pred = output.argmax(dim=1, keepdim=True)
                    disc_acc = accuracy(pred, labels)
                    if tb_writer:
                        tb_writer.add_scalar("Accuracy/Train", disc_acc, global_step)

                    with torch.no_grad():
                        gen_imgs = []
                        for i, img in enumerate(artif_imgs):
                            if img.device != "cpu":
                                img = img.cpu()
                            img = decode_img(img.detach().numpy())
                            img = array_yxc2cyx(img)
                            gen_imgs.append(torch.from_numpy(img))
                        if tb_writer:
                            img_grid_gen = torchvision.utils.make_grid(gen_imgs)
                            tb_writer.add_image('Valid/Generated', img_tensor=img_grid_gen,
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
    parser.add_argument("--device", type=str, default="cpu",
                        help="Target device: cpu/cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
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

    dataset = FacesDataset("/home/igor/datasets/faces6k/aligned", target_size=(64, 64),
                           mean=DEFAULT_RGB_MEAN, std=DEFAULT_RGB_STD)
    train_dloader = make_dataloader(dataset, batch_size=args.batch_train, shuffle_dataset=True)

    params = list(generator.parameters()) + list(discriminator.parameters())
    optimizer = None
    if args.optimizer == "adam":
        optimizer = Adam(params, lr=args.learning_rate,
                         weight_decay=args.l2)
    elif args.optimizer == "sgd":
        optimizer = SGD(params, lr=args.learning_rate,
                        weight_decay=args.l2)

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    tboard_writer = SummaryWriter()

    try:
        for e in range(args.epochs):
            train(generator, discriminator, train_dloader, optimizer, e, scheduler=scheduler, device=args.device,
                  autosave_period=None, valid_period=args.valid_period, tb_writer=tboard_writer)
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

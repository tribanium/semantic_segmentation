from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from unet import UNet

img_dir = "./data/train/images/"
masks_dir = "./data/train/masks/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TGSDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.all_images = os.listdir(img_path)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        file_name = self.all_images[index]
        input_img = (
            Image.open(os.path.join(self.img_path, file_name))
            .convert("L")
            .resize((128, 128))
        )
        mask_img = (
            Image.open(os.path.join(self.mask_path, file_name))
            .convert("L")
            .resize((128, 128))
        )
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_img = self.transform(mask_img)
        return input_img, mask_img


class Model:
    def __init__(self):
        self.num_epochs = 50
        self.batch_size = 32

        self.input_size = (128, 128)

        self.net = UNet().to(device)
        self.optimizer = optim.Adam(self.net.parameters())
        self.loss = nn.BCELoss()

        self.isSchedule = False
        if self.isSchedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min"
            )

        transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = TGSDataset(img_dir, masks_dir, transform=transform)

        self.train_loss = []
        self.test_loss = []
        self.save_loss = True

        test_split = 0.2
        shuffle_dataset = True
        random_seed = 42
        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        self.test_dataloader = DataLoader(
            self.dataset, batch_size=1, sampler=test_sampler
        )

    def train(self):
        print("Training model...\n")
        best_loss = 1e99
        for epoch in range(self.num_epochs):
            print(15 * "=" + f" Epoch {epoch+1}/{self.num_epochs} " + 15 * "=")
            train_loss_epoch = []
            test_loss_epoch = []

            for _, (img, mask) in enumerate(tqdm(self.train_dataloader)):

                img, mask = img.to(device), mask.to(device)

                self.optimizer.zero_grad()

                y_pred = self.net(img)
                loss = self.loss(y_pred, mask)
                loss.backward()
                self.optimizer.step()

                train_loss_epoch.append(loss.item())

            avg_train_loss = np.mean(train_loss_epoch)
            print(f"Average train loss\t:\t{avg_train_loss}")
            self.train_loss.append(avg_train_loss)

            with torch.no_grad():
                for _, (img, mask) in enumerate(self.test_dataloader):
                    img, mask = img.to(device), mask.to(device)

                    self.optimizer.zero_grad()

                    y_pred = self.net(img)
                    loss = self.loss(y_pred, mask)

                    test_loss_epoch.append(loss.item())

                avg_test_loss = np.mean(test_loss_epoch)
                print(f"Average test loss\t:\t{avg_test_loss}\n")
                self.test_loss.append(avg_test_loss)
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    print("***\tSaving best model on average test loss\t***\n")
                    torch.save(self.net.state_dict(), "./model.pt")

        if self.save_loss:
            self.save_loss_pkl()

    def load_model(self):
        self.net.load_state_dict(
            torch.load("./model.pt", map_location=torch.device("cpu"))
        )

    def infer_data(self):
        f, axarr = plt.subplots(6, 3)
        with torch.no_grad():
            for k in range(0, 6):
                _, (img_batch, mask_batch) = next(enumerate(self.test_dataloader))
                pred = self.net(img_batch)
                img = img_batch.numpy().squeeze(axis=(0, 1))
                mask = mask_batch.numpy().squeeze(axis=(0, 1))
                pred = pred.numpy().squeeze(axis=(0, 1))
                axarr[k, 0].imshow(img)
                axarr[k, 1].imshow(mask)
                axarr[k, 2].imshow(pred)
                axarr[k, 0].set_axis_off()
                axarr[k, 1].set_axis_off()
                axarr[k, 2].set_axis_off()
            axarr[0, 0].set_title("Input")
            axarr[0, 1].set_title("Ground truth")
            axarr[0, 2].set_title("Pred")
            plt.show()

    def plot_sample(self):
        f, axarr = plt.subplots(2, 4)
        nb_samples = 4
        for k in range(0, nb_samples):
            image, mask = self.dataset[np.random.randint(0, len(self.dataset))]
            axarr[0, k].imshow(
                image.squeeze(axis=0),
            )
            axarr[1, k].imshow(mask.squeeze(axis=0))
            axarr[0, k].set_axis_off()
            axarr[1, k].set_axis_off()
        plt.show()

    def save_loss_pkl(self):
        print("Saving train and test loss...")
        with open("train_loss.pkl", "wb") as fp:
            pickle.dump(self.train_loss, fp)

        with open("test_loss.pkl", "wb") as fp:
            pickle.dump(self.test_loss, fp)

    def load_loss_pkl(self):
        with open("train_loss.pkl", "rb") as fp:
            self.train_loss = pickle.load(fp)

        with open("test_loss.pkl", "rb") as fp:
            self.test_loss = pickle.load(fp)

    def plot_curves(self):
        x = [i for i in range(1, self.num_epochs + 1)]
        plt.plot(x, self.train_loss, label="Train")
        plt.plot(x, self.test_loss, label="Test", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Binary Cross-Entropy")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Runs the training loop", action="store_true")
    parser.add_argument(
        "--infer", help="Infers results on the test set", action="store_true"
    )
    parser.add_argument(
        "--curves", help="Plots the learning curves", action="store_true"
    )
    args = parser.parse_args()

    model = Model()

    if args.train:
        model.train()

    elif args.infer:
        model.load_model()
        model.infer_data()

    elif args.curves:
        model.load_loss_pkl()
        print(f"Min test loss : {min(model.test_loss)}")
        model.plot_curves()

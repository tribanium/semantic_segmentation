from pickletools import uint8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from glob import glob

from unet import UNet
from unet_no_pool import UNetNoPool
from customdataset import TGSDataset

img_dir = "./data/train/images/"
masks_dir = "./data/train/masks/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, no_pool=False):
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-3

        # Stops the training if there's no improvement of test loss after this number of epochs
        self.early_stopping_epochs = 10

        # Instanciating the model, optimizer and loss
        if no_pool:
            self.net = UNetNoPool().to(device)
        else:
            self.net = UNet().to(device)

        self.optimizer = optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )  # Default lr=1e-3
        self.loss = nn.BCELoss()  # Binary cross-entropy

        # Learning rate decay
        self.isSchedule = True
        if self.isSchedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                patience=5,
                factor=0.1,
                min_lr=1e-5,
                verbose=True,
            )

        # Allows to save the loss of train and test set during training
        self.train_loss = []
        self.test_loss = []
        self.save_loss = True

        # Creating the dataset from images folders
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = TGSDataset(img_dir, masks_dir, transform=transform)

        # Splitting the 4000 items dataset into a train and test set with 80/20 ratio
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
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        self.train_dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        self.test_dataloader = DataLoader(
            self.dataset, batch_size=1, sampler=test_sampler
        )

    def train(self):
        """
        Training loop.
        """

        print(f"\nTraining model : {self.net.__class__.__name__}...\n")
        print(
            f"learning rate = {self.learning_rate}\tbatch size = {self.batch_size}\t early stopping : {self.early_stopping_epochs} epochs\n"
        )
        if self.isSchedule:
            print("LR scheduler is on.\n")
        else:
            print("LR scheduler is off.\n")

        best_loss = 1e99
        early_stopping_counter = 0

        for epoch in range(self.num_epochs):
            # Early stopping
            if early_stopping_counter == self.early_stopping_epochs:
                print(
                    f"#####\tNo improvement of test loss since {self.early_stopping_epochs} epochs. Stopping the training.\t#####"
                )
                break

            print(15 * "=" + f" Epoch {epoch+1}/{self.num_epochs} " + 15 * "=")

            # Contains the loss for each batch during an epoch
            train_loss_epoch = []
            test_loss_epoch = []

            for _, (img, mask) in enumerate(tqdm(self.train_dataloader)):
                # Useful when training on GPU
                img, mask = img.to(device), mask.to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass and backpropagation
                y_pred = self.net(img)
                loss = self.loss(y_pred, mask)
                loss.backward()
                self.optimizer.step()

                train_loss_epoch.append(loss.item())

            avg_train_loss = np.mean(train_loss_epoch)
            print(f"Average train loss\t:\t{avg_train_loss}")
            self.train_loss.append(avg_train_loss)

            # Don't take the forward pass into account here
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
                # Saving the model and resetting the early stopping counter if there is a test loss improvement
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    print("*****\tSaving best model on average test loss\t*****\n")
                    torch.save(self.net.state_dict(), "./model.pt")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(
                        f"#####\t{early_stopping_counter} epochs since last improvement\t#####\n"
                    )
            if self.isSchedule:
                self.scheduler.step(avg_test_loss)

        # Saving the train and test loss to plot learning curves later
        if self.save_loss:
            self.save_loss_pkl()

        print("\nDone !\n")

    def load_model(self, dir="."):
        """
        Loads a trained model.
        """
        model_path = f"{dir}/model.pt"
        self.net.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )

    def infer_data(self, threshold=0.5):
        """
        Plots a 6x4 grid containing 6 test samples with :
        - the input
        - the ground truth mask
        - the prediction
        - the binary prediction with a given threshold (default = 0.5)
        """

        f, axarr = plt.subplots(6, 4)
        with torch.no_grad():
            for k in range(0, 6):
                _, (img_batch, mask_batch) = next(enumerate(self.test_dataloader))
                pred = self.net(img_batch)
                img = img_batch.numpy().squeeze(axis=(0, 1))
                mask = mask_batch.numpy().squeeze(axis=(0, 1))
                pred = pred.numpy().squeeze(axis=(0, 1))
                pred_bin = pred > threshold
                axarr[k, 0].imshow(img)
                axarr[k, 1].imshow(mask)
                axarr[k, 2].imshow(pred)
                axarr[k, 3].imshow(pred_bin)
                axarr[k, 0].set_axis_off()
                axarr[k, 1].set_axis_off()
                axarr[k, 2].set_axis_off()
                axarr[k, 3].set_axis_off()
            axarr[0, 0].set_title("Input")
            axarr[0, 1].set_title("Ground truth")
            axarr[0, 2].set_title("Pred")
            axarr[0, 3].set_title("Pred binary")
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
        """
        Saves the train_loss and test_loss lists as .pkl files
        """

        print("Saving train and test loss...")
        with open("train_loss.pkl", "wb") as fp:
            pickle.dump(self.train_loss, fp)

        with open("test_loss.pkl", "wb") as fp:
            pickle.dump(self.test_loss, fp)

    def load_loss_pkl(self, dir="."):
        """
        Loads the train_loss and test_loss lists from .pkl files
        """
        train_path = f"{dir}/train_loss.pkl"
        test_path = f"{dir}/test_loss.pkl"

        with open(train_path, "rb") as fp:
            self.train_loss = pickle.load(fp)

        with open(test_path, "rb") as fp:
            self.test_loss = pickle.load(fp)

    def plot_curves(self):
        """
        Plots the learning curves.
        """
        index_min = np.argmin(self.test_loss) + 1
        min_loss = min(self.test_loss)
        print(f"Min test loss : {min_loss}")
        x = [i for i in range(1, len(self.train_loss) + 1)]
        plt.plot(x, self.train_loss, label="train")
        plt.plot(x, self.test_loss, label="test", color="red")
        plt.axvline(
            x=index_min,
            color="black",
            ls="--",
            label=f"min test : {'{:.4f}'.format(min_loss)}",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Binary Cross-Entropy")
        plt.legend(loc="upper right", framealpha=1.0, prop={"size": 9})
        plt.show()

    def compute_roc(self):
        """Computes true positives rate and false positives rate for different thresholds and stores the results in a dict then saves it into a .pkl file"""
        length_images = 128 * 128
        # false positive rate
        fpr = []
        # true positive rate
        tpr = []
        # Iterate thresholds from 0.0, 0.01, ... 1.0
        thresholds = np.arange(0.0, 1.1, 0.1)
        with torch.no_grad():
            for threshold in tqdm(thresholds):
                FP = 0
                TP = 0
                for _, (img, mask) in enumerate(tqdm(self.test_dataloader)):
                    pred = self.net(img).numpy().ravel() > threshold
                    mask = mask.numpy().ravel() > 0.5
                    TP += np.sum(pred & mask) / length_images
                    FP += np.sum((pred == 1) & (mask == 0)) / length_images
                fpr.append(FP / len(self.test_dataloader))
                tpr.append(TP / len(self.test_dataloader))

        dico = {"fpr": fpr, "tpr": tpr}
        with open("roc.pkl", "wb") as fp:
            pickle.dump(dico, fp)

    def plot_ROC(self):
        """Plots the ROC curve. Checks if the pkl files exists first. If not, it computes it."""
        is_roc = [y for x in os.walk("./") for y in glob(os.path.join(x[0], "roc.pkl"))]
        if is_roc:
            with open(is_roc[0], "rb") as fp:
                dico = pickle.load(fp)

            fpr = dico["fpr"]
            tpr = dico["tpr"]
            plt.plot(fpr, tpr, "x-", ms=4)
            plt.plot(fpr, fpr, "k--")
            plt.xlabel("False positives")
            plt.ylabel("True positives")
            plt.title("ROC curve")
            plt.show()
        else:
            self.compute_roc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Runs the training loop", action="store_true")
    parser.add_argument(
        "--train_no_pool",
        help="Trains the Unet without pooling layers. Use it with --train flag",
        action="store_true",
    )
    parser.add_argument(
        "--infer", help="Infers results on the test set", action="store_true"
    )
    parser.add_argument(
        "--curves", help="Plots the learning curves", action="store_true"
    )
    parser.add_argument("--roc", help="Plots the roc", action="store_true")
    parser.add_argument(
        "--dir",
        help="Directory in which the model and the pkl files are stored",
        default=".",
    )
    args = parser.parse_args()

    if args.train_no_pool:
        # Trains the model without maxpooling layers
        model = Model(no_pool=True)
        model.train()

    else:
        model = Model()

        if args.train:
            model.train()

        elif args.infer:
            model.load_model(dir=args.dir)
            model.infer_data()

        elif args.curves:
            model.load_loss_pkl(dir=args.dir)
            model.plot_curves()

        elif args.roc:
            model.load_model(dir=args.dir)
            model.plot_ROC()

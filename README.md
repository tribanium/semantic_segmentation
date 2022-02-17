# Semantic segmentation
The goal of this lab is to train a [UNet model](https://arxiv.org/pdf/1505.04597.pdf) with an autoencoder structure to segment images obtained by seismic reflection and to detect the presence of salt or sediments in the sea bed.



## Downloading the data
The data was downloaded on https://www.kaggle.com/c/tgs-salt-identification-challenge/data?select=train.zip. It is composed of an `images` and `masks` folders, containing 4000 `.png` files of size 101x101 (that we resize at 128x128).

The data was then splitted in a train (80%) and test set (20%). 

## Getting started
To train the model :
```
python main.py --train
```
or to train it with the UNetNoPool (where max pooling layers where removed and convolutions have `stride=2` and `padding=1`) :
```
python main.py --train_no_pool
```


## Visualize the results
Once the UNet is trained, the file `model.pt` contains the trained model with the lowest loss obtained on the test set. The `train_loss` and `test_loss` are also stored as `.pkl` files.


You can infer a sample of test images with :

```
python main.py --infer
```

The files `train_loss.pkl` and `test_loss.pkl` allow you to plot the learning curves with
```
python main.py --curves
```

You can also plot the ROC curve with :

```
python main.py --roc
```
**Note :** the `roc.pkl` file is not inside the root directory or a subfolder, it will be computed (it's a bit long !)




Assuming your `model.pt` or `.pkl` files are stored into `./example/` directory, you can type :
```
python main.py --curves --dir example
python main.py --infer --dir example
python main.py --roc --dir example
```

## Results
### Learning curves
We chose to keep the model (`model.pt`) with the lowest loss on the test set : here obtained at epoch 32, with `test_loss = 0.1784`.

![](img/curves.png)

### Inference of test samples
We chose a `threshold = 0.5` when binarizing the predictions :

![](img/inference.png)

### Receiver Operating Characteristic (ROC) curve
This curve show the TP rate VS FP rate. The rightmost point corresponds to a 0 threshold (everything is classified to 1) and increments 0.1 when going left.

![](img/roc_curve.png)
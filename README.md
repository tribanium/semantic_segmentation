# Semantic segmentation

## Getting started
To train the model :
```
python main.py --train
```

Once the UNet is trained, the file `model.pt` contains the trained model with the lowest loss obtained on the test set.


As the model was already trained, you can infer a sample of test images with 

```
python main.py --infer
```

The files `train_loss.pkl` and `test_loss.pkl` allow you to plot the learning curves with
```
python main.py --curves
```
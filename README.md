# vistamilk_diet_challenge
Repository for the 2022 VistaMilk spectroscopy challenge.

## Notebooks
The notebooks are, for the moment, preliminary.

## `exploration.ipynb`
This contains some initial considerations and views on the data. A number of
models can be trained here, but no optimization is currently done.

## `image_classification.ipynb`
This contains the basic steps for a CNN that would take as input the series
shaped as images. The network structure is just tentative.

If you want to look at the real-time network performance, create a folder called
`logs`, and when the network is training, run this in your terminal:

```
tensorboard --logdir logs/fit
```

## Running the scripts
Please run the deep model scripts from the root directory of the project. They
will not work if not invoked as python modules.

### Simple training: `deep_model`
You can train a deep model with the `deep_model.py` script. This will train
a model using a single 60/40 split.

```bash
$ python -m deep_models.deep_model --name my_model --arch fcn --data no_water
```

The arguments of the script are the following:

- `--name`, `-n`: the name of the model to train, will be used to store training
  history and the serialized model.
- `--arch`, `-a`: the model architecture, can be `fcn` for the full connected
  network, `cnn` for the convolutional network with compact filters, or
  `cnn_dilated` for the convolutional network with dilated filters.
- `--data`, `-d`: data modality, can be `full` if all the wave components should
  be used, or `no_water` if water regions should be excluded.

The training history will be saved in the `./logs/fit/$NAME` folder. You can
use `tensorboard` to explore it by running the command:

```bash
$ tensorboard --logdir logs/fit
```

Once the server runs, simply visit `https://localhost:6006` to see it.

After the training is complete, the trained model is then stored in
`models/$NAME` alongside a json file containing the training history.

### Cross-validation: `deep_model_cv`
If you want to train a model using 3-fold CV, just execute this one script.
Arguments are the same as `deep_model`, but training history and trained models
will not be stored in this case. However, the accuracy scores of the 3 splits
will be saved in a text file contained in the `cv/$NAME.txt` file.

### Best model: `train_best.py`
This script works like the previous 2, but in this case no validation is
performed and there is no early stopping. In short, the model produced for
the submission was obtained with:

```bash
$ python -m deep_models.train_best --name best --arch fcn --data no_water
```
This script also loads the test dataset, transforms it using the same pipeline
applied to the training dataset, and generate the predctions, storing them
in the `data/raw_test_predictions.csv` file.

# Keras (TensorFlow) implementation of RaGAN.
  
This is an implementation of a Relativistic Average GAN, introduced in ["The relativistic discriminator: a key element missing from standard GAN".](https://arxiv.org/abs/1807.0073)
The original Pytorch code is found [here](https://github.com/AlexiaJM/RelativisticGAN).

## Usage

  1. Download the (Fashion)-MNIST *.csv dataset from kaggle, and place in `./data/fashion-mnist_train.csv`: 
    - https://www.kaggle.com/oddrationale/mnist-in-csv
    - https://www.kaggle.com/zalando-research/fashionmnist/data
  2. Create the Logdir: `$ mkdir ./logs/` .
  3. Run `$ python ragan.py` to train using the default hyperparameters, or run `$ python ragan.py -h` for information about the tweakable parameters.

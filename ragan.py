import matplotlib as mpl
mpl.use("Agg")

# General imports
import numpy as np
from time import time
import matplotlib.pyplot as plt
import argparse

# ML specific imports
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, LeakyReLU, GlobalAveragePooling2D, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

def get_images(file):
    """
    Generate numpy array from MNIST/Fashion-MNIST images for training RaGAN.
    """
    ims = np.genfromtxt(file, delimiter=",", skip_header=1)[:,1:] # Don't want the label
    ims = np.reshape(ims, [-1,28,28,1])
    ims = ims/255*2-1 # Normalise
    return ims

def gen(z_shape=(100,)):
    """
    Model a standard DCGAN generator.
    """
    z = Input(shape=z_shape)

    d0 = Dense(7*7*128, activation="relu")(z)
    d0 = Reshape([7,7,128])(d0)

    ct0 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same", activation="relu")(d0)
    ct0 = BatchNormalization(momentum=0.9, epsilon=0.00002)(ct0)

    ct1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", activation="relu")(ct0)
    ct1 = BatchNormalization(momentum=0.9, epsilon=0.00002)(ct1)

    G_z = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same", activation="tanh")(ct1)
    model = Model(z, G_z, name="Generator")
    model.summary()
    return model

def disc(x_shape=(28,28,1)):
    """
    Model a standard DCGAN discriminator.
    """
    x = Input(shape=x_shape)

    c0 = Conv2D(filters=32, kernel_size=4, strides=2, padding="same")(x)
    c0 = LeakyReLU(0.1)(c0)

    c1 = Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(c0)
    c1 = LeakyReLU(0.1)(c1)

    c2 = Conv2D(filters=128, kernel_size=4, strides=2, padding="same")(c1)
    c2 = LeakyReLU(0.1)(c2)

    gap = GlobalAveragePooling2D()(c2)
    y = Dense(1)(gap) # binary output with no one-hot encoding
    model = Model(x, y, name="Discriminator")
    model.summary()
    return model

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Run RaGAN on MNIST/Fashion-MNIST data.")
    # Args
    parser.add_argument("-f", "--im_file", type=argparse.FileType('r'), default="./data/fashion-mnist_train.csv", help="*.csv file containing (Fashion) MNIST Image data.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size, default 64.")
    parser.add_argument("-t", "--train_ratio", type=int, default=1, help="Gen/Disc training ratio (how many batches to train Gen on before training Disc once), default 1.")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of training epochs, default 300.")
    parser.add_argument("-l", "--logdir", nargs="?", default="./logs", dest="logdir", help="Logdir, default ./logs")
    args = parser.parse_args()

    batch_size = args.batch_size
    train_ratio = args.train_ratio
    epochs = args.epochs
    logdir = args.logdir
    test_batch_size = 100

    adam_op = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    ims = get_images(args.im_file)

    # Define generator and discriminator models
    gen = gen()
    disc = disc()

    # Define real and fake images
    reals = Input(shape=ims.shape[1:])
    z = Input(shape=(100,))
    fakes = gen(z)
    disc_r = disc(reals) # C(x_r)
    disc_f = disc(fakes) # C(x_f)

    # Define generator and discriminator losses according to RaGAN described in Jolicoeur-Martineau (2018).
    # Dummy predictions and trues are needed in Keras.
    def rel_disc_loss(y_true, y_pred):
        epsilon=0.000001
        return -(K.mean(K.log(K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)\
                 +K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0))

    def rel_gen_loss(y_true, y_pred):
        epsilon=0.000001
        return -(K.mean(K.log(K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)\
                 +K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0))

    # Define trainable generator and discriminator
    gen_train = Model([z, reals], [disc_r, disc_f])
    disc.trainable = False
    gen_train.compile(adam_op, loss=[rel_gen_loss, None])
    gen_train.summary()

    disc_train = Model([z, reals], [disc_r, disc_f])
    gen.trainable = False
    disc.trainable = True
    disc_train.compile(adam_op, loss=[rel_disc_loss, None])
    disc_train.summary()

    # Train RaGAN
    gen_loss = []
    disc_loss = []

    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    test_z = np.random.randn(test_batch_size, 100).astype(np.float32)

    start_time = time()
    for epoch in np.arange(epochs):
        np.random.shuffle(ims)

        print(epoch,"/",epochs)
        n_batches = int(len(ims) // batch_size)
        minibatch_size = batch_size * (train_ratio+1) # why??

        prog_bar = Progbar(target=int(len(ims) // minibatch_size))
        batch_start_time = time()

        for index in np.arange(int(len(ims) // minibatch_size)):
            prog_bar.update(index)
            it_minibatch = ims[index*minibatch_size:(index+1)*minibatch_size]

            for j in np.arange(train_ratio): # might want to switch disc/gen
                image_batch = it_minibatch[j*batch_size:(j+1)*batch_size]
                z = np.random.randn(batch_size, 100).astype(np.float32)
                disc.trainable = False
                gen.trainable=True
                gen_loss.append(gen_train.train_on_batch([z, image_batch], dummy_y))

            image_batch = it_minibatch[train_ratio*batch_size:(train_ratio+1)*batch_size]
            z = np.random.randn(batch_size, 100).astype(np.float32)
            disc.trainable = True
            gen.trainable = False
            disc_loss.append(disc_train.train_on_batch([z, image_batch], dummy_y))


        print("\nBatch time", int(time()-batch_start_time))
        print("Total elapsed time", int(time()-start_time))

        ## Print out losses and pics of G(z) outputs ##

        gen_image = gen.predict(test_z)
        fig, axs = plt.subplots(nrows=int(np.sqrt(test_batch_size)), ncols=int(np.sqrt(test_batch_size)), figsize=(8, 8))
        for i, ax in enumerate(axs.ravel()):
            ax.axis("off")
            ax.imshow(gen_image[i,...,0], cmap="gray")
        plt.savefig(logdir+"/"+str(int(batch_start_time))+"-ex-epoch-"+str(epoch)+".png")
        plt.close(fig)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
        disc_loss_ar = np.array(disc_loss)[:,0]
        gen_loss_ar = np.array(gen_loss)[:,0]
        axs.set_title("Losses at epoch "+str(epoch))
        axs.set_xlabel("Global step")
        axs.set_ylabel("Loss")
        axs.plot(disc_loss_ar, label="disc loss")
        axs.plot(gen_loss_ar, label="gen loss")
        axs.legend()
        plt.savefig(logdir+"/"+str(int(batch_start_time))+"-loss-epoch-"+str(epoch)+".png")
        plt.close(fig)

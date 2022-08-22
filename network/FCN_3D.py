import os
import time
import h5py
import pandas as pd
import numpy as np 
import tensorflow as tf
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from .Unet_3D import unet_model_3d, unet_model_3d_maxPooling


class SR_UnetGAN_3D:
    def __init__(
        self, 
        data_path=None,
        train_data_len=None,
        val_data_len=None,
        save_dir='FCN_3D',
        img_shape=(64,64,64),
        epochs=10,
        batch_size=4,
        loss='mse',
        activation=None,
        max_pooling=False
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.img_shape = img_shape
        self.common_optimizer = Adam(0.0002, 0.5)
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_len = train_data_len
        self.validation_data_len = val_data_len
        self.activation = activation
        self.max_pooling = max_pooling
        self.generator = self.build_generator()
        self.generator.compile(loss=loss, optimizer=self.common_optimizer)

    def build_generator(self):
        if self.max_pooling:
            model = unet_model_3d_maxPooling(self.img_shape, activation=self.activation)
        else:
            model = unet_model_3d(self.img_shape, activation=self.activation)
        return model

    def data_generator(self, data, name, length):
        while True:
            x_batch, y_batch = [], []
            count = 0
            for j in range(length):
                x = data.get(name+'_noisy')[j]
                y = data.get(name+'_original')[j]
                if self.img_shape[0]==1:
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                x_batch.append(x)
                y_batch.append(y)
                count += 1
                if count == self.batch_size:
                    yield np.array(x_batch), np.array(y_batch)
                    count = 0
                    x_batch, y_batch = [], []

    def write_log(self, callback, name, value, batch_no):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

    def train(self):
        """Subject FCN_3D training job, models and logs are automatically saved in output dir."""
        data = h5py.File(self.data_path, mode='r')
        train_data_generator = self.data_generator(data, 'train', self.train_data_len)
        validation_data_generator = self.data_generator(data, 'validation', self.validation_data_len)

        log_dir = os.path.join(self.save_dir, 'log')
        tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.asctime()))
        tensorboard.set_model(self.generator)
        batch_df = pd.DataFrame()
        epoch_df = pd.DataFrame()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)

            model_dir = os.path.join(self.save_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Add a loop, which will run for a specified number of epochs:
            for epoch in range(1, self.epochs+1):
                # Create two lists to store losses
                gen_losses, val_losses = [], []
                number_of_batches = int(self.train_data_len / self.batch_size)
                for index in range(number_of_batches):
                    print('Epoch: {}/{}\n  Batch: {}/{}'.format(epoch, self.epochs, index+1, number_of_batches))
                    
                    train_x_batch, train_y_batch = next(train_data_generator)
                    g_loss = self.generator.train_on_batch(train_x_batch, train_y_batch)
                    gen_losses.append(g_loss)
                    print("    G_loss: {}\n".format(g_loss))

                    val_x_batch, val_y_batch = next(validation_data_generator)
                    v_loss = self.generator.test_on_batch(val_x_batch, val_y_batch)
                    val_losses.append(v_loss)
                    print("    V_loss: {}\n".format(v_loss))

                batch_df = batch_df.append(pd.DataFrame({'epoch': [epoch] * len(gen_losses),
                                                         'batch': np.arange(1, len(gen_losses)+1),
                                                         'generator_loss': gen_losses,
                                                         'validation_loss': val_losses}))
                epoch_df = epoch_df.append(pd.DataFrame({'epoch': [epoch],
                                                         'generator_loss': [np.mean(gen_losses)],
                                                         'validation_loss': [np.mean(val_losses)]}))
                batch_df = batch_df[['epoch', 'batch', 'generator_loss', 'validation_loss']]
                epoch_df = epoch_df[['epoch', 'generator_loss', 'validation_loss']]
                batch_df.to_csv(os.path.join(log_dir, 'batch_loss.csv'), index=False)
                epoch_df.to_csv(os.path.join(log_dir, 'epoch_loss.csv'), index=False)
                # Save losses to Tensorboard
                self.write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)
                self.write_log(tensorboard, 'validation_loss', np.mean(val_losses), epoch)

                # Save model at each epoch
                self.generator.save_weights(os.path.join(model_dir, 'generator_epoch_{}.hdf5'.format(epoch)))
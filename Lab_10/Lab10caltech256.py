#!/usr/bin/env python
"""
Fresh/Rotten fruit predictor

Written by Derek Riley, inspired by Nvidia DLI training

# Building a model
# --data - subdirectory of images for training
# --batch_size - batch size to use for training
# --epochs - amount of epochs to use for training
# --main_dir - where to save produced models, defaults to working directory
# --augment_data - boolean indication for whether to use data augmentation
# --fine_tune - boolean indication for whether to use fine tuning

Note:
    - directory arguments must not be followed by a '/'
        Good: home/username
        Bad: home/username/
    
Example:
    
    python Lab9.py --data /data/cs2300/L9/fruits --batch_size 32 --epochs 10 --main_dir home/<username> --augment_data false --fine_tune true

"""

import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import horovod.keras as hvd
from keras import backend as K
import keras.optimizers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--main_dir', type=str, default='')
    parser.add_argument('--augment_data', type=str, default='')
    parser.add_argument('--fine_tune', type=str, default='')
    return parser.parse_args()

def main():
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))


    # start by parsing the command line arguments 
    args = parse_args()
    data = args.data
    my_batch_size = int(args.batch_size)
    my_epochs = int(args.epochs)
    augment_data = args.augment_data
    fine_tune = args.fine_tune
    h5modeloutput = 'model_b' + args.batch_size + '_e' + args.epochs + '_aug' + \
        args.augment_data + '_ft' + args.fine_tune + '.h5'
    print(args)
    
    # Load weights pre-trained on the ImageNet model
    base_model = keras.applications.VGG16(
        weights='imagenet',  
        input_shape=(224, 224, 3),
        include_top=False)

    # Next, we will freeze the base model so that all the learning from the ImageNet 
    # dataset does not get destroyed in the initial training.
    base_model.trainable = False

    # Create inputs with correct shape
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x.trainable = False

    # Add pooling layer or flatten layer
    x =  keras.layers.GlobalAveragePooling2D()(x)

    # Add final dense layer with 6 classes for the 6 types of fruit
    outputs = keras.layers.Dense(257, activation = 'softmax')(x)

    # Combine inputs and outputs to create model
    model = keras.Model(inputs, outputs)
    
    opt = keras.optimizers.Adam(lr=1.0*hvd.size())
    opt = hvd.DistributedOptimizer(opt)


    # uncomment the following code if you want to see the model
    # model.summary()

    # Now it's time to compile the model with loss and metrics options. 
    model.compile(optimizer=opt, loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    datagen = ImageDataGenerator(
            samplewise_center=True,  # set each sample mean to 0
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0, # randomly zoom image 
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            validation_split=0.2, # added for caltech256 since it doesn't have pre-split data
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False) # randomly flip images

    # These are data augmentation steps
    if(augment_data.lower() in ['true', '1', 't', 'y', 'yes']):
        datagen = ImageDataGenerator(
                samplewise_center=True,  # set each sample mean to 0
                rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # randomly zoom image 
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                validation_split=0.2, # added for caltech256 since it doesn't have pre-split data
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False) # randomly flip images

    # load and iterate training dataset
    train_it = datagen.flow_from_directory( data,# + '/train/', 
                                           target_size=(224,224), 
                                           color_mode='rgb', 
                                           batch_size=my_batch_size,
                                           subset='training', # added for caltech256 since it doesn't have pre-split data
                                           class_mode="categorical")
    # load and iterate validation dataset
    valid_it = datagen.flow_from_directory( data,# + '/test/', 
                                          target_size=(224,224), 
                                          color_mode='rgb', 
                                          batch_size=my_batch_size,
                                          subset='validation', # added for caltech256 since it doesn't have pre-split data
                                          class_mode="categorical")

    #Broadcast the initial state
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    # Train the model
    history_object = model.fit(train_it,
              validation_data=valid_it,
              steps_per_epoch=train_it.samples/train_it.batch_size//hvd.size(),
              validation_steps=valid_it.samples/valid_it.batch_size//hvd.size(),
              epochs=my_epochs,
              callbacks=callbacks, 
              verbose=2)

    if(fine_tune.lower() in ['true', '1', 't', 'y', 'yes']):
        print("fine tuning...")
        # This will improve the accuracy of the model by fine tuning the training on the entire unfrozen model.  
        # Unfreeze the base model
        base_model.trainable = True
        
        opt_ft = keras.optimizers.RMSprop(learning_rate = .000001*hvd.size())
        opt_ft = hvd.DistributedOptimizer(opt_ft)

        
        # Compile the model with a low learning rate
        model.compile(optimizer=opt_ft,
                      loss ='categorical_crossentropy' , metrics = ['accuracy'])

        history_object = model.fit(train_it,
                  validation_data=valid_it,
                  steps_per_epoch=train_it.samples/train_it.batch_size//hvd.size(),
                  validation_steps=valid_it.samples/valid_it.batch_size//hvd.size(),
                  epochs=my_epochs,
                  callbacks=callbacks, 
                  verbose=2)
                  
    save_loss_plot(history_object.history, args)


def save_loss_plot(history, args):
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Fruit Classification, Batch Size: ' + args.batch_size + ' Epochs: ' + args.epochs)
    plt.legend()
    plt.savefig(args.main_dir + '/' + 'model_b' + args.batch_size + '_e' + args.epochs + ' rank ' + str(hvd.rank()) + '.png')
    print(args.main_dir + '/' + 'model_b' + args.batch_size + '_e' + args.epochs + ' rank ' + str(hvd.rank()) + '.png')

if __name__ == "__main__":
    main()


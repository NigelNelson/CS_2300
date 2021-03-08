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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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
    x = base_model(inputs, training=False)

    # Add pooling layer or flatten layer
    x =  keras.layers.GlobalAveragePooling2D()(x)

    # Add final dense layer with 6 classes for the 6 types of fruit
    outputs = keras.layers.Dense(6, activation = 'softmax')(x)

    # Combine inputs and outputs to create model
    model = keras.Model(inputs, outputs)

    # uncomment the following code if you want to see the model
    # model.summary()

    # Now it's time to compile the model with loss and metrics options. 
    model.compile(optimizer='Adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    datagen = ImageDataGenerator(
            samplewise_center=True,  # set each sample mean to 0
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0, # randomly zoom image 
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False) # randomly flip images

    # These are data augmentation steps
    if(augment_data.lower() in ['true', '1', 't', 'y', 'yes']):
        datagen = ImageDataGenerator(
                samplewise_center=True,  # set each sample mean to 0
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.1, # randomly zoom image 
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False) # randomly flip images

    # load and iterate training dataset
    train_it = datagen.flow_from_directory( data + '/train/', 
                                           target_size=(224,224), 
                                           color_mode='rgb', 
                                           batch_size=my_batch_size,
                                           class_mode="categorical")
    # load and iterate validation dataset
    valid_it = datagen.flow_from_directory( data + '/test/', 
                                          target_size=(224,224), 
                                          color_mode='rgb', 
                                           batch_size=my_batch_size,
                                          class_mode="categorical")

    # Train the model
    history_object = model.fit(train_it,
              validation_data=valid_it,
              steps_per_epoch=train_it.samples/train_it.batch_size,
              validation_steps=valid_it.samples/valid_it.batch_size,
              epochs=my_epochs)

    if(fine_tune.lower() in ['true', '1', 't', 'y', 'yes']):
        # This will improve the accuracy of the model by fine tuning the training on the entire unfrozen model.  
        # Unfreeze the base model
        base_model.trainable = True
        # Compile the model with a low learning rate
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
                      loss =  'categorical_crossentropy' , metrics = ['accuracy'])

        history_object = model.fit(train_it,
                  validation_data=valid_it,
                  steps_per_epoch=train_it.samples/train_it.batch_size,
                  validation_steps=valid_it.samples/valid_it.batch_size,
                  epochs=my_epochs)
                  
    save_loss_plot(history_object.history, args)
    model.save(args.main_dir + '/' + h5modeloutput)


def save_loss_plot(history, args):
    plt.plot(history['acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Fruit Classification, Batch Size: ' + args.batch_size + ' Epochs: ' + args.epochs)
    plt.legend()
    plt.savefig(args.main_dir + '/' + 'model_b' + args.batch_size + '_e' + args.epochs + '.png')

if __name__ == "__main__":
    main()


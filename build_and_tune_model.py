from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from encoder import *
from pairing import *

## GLOBAL VARIABLES
import os
# specify the shape of the inputs for our network
IMG_SHAPE = (25, 25, 1)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 300
# define the path to the base output directory
BASE_OUTPUT = "output"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

LOG_DIR = f"{int(time.time())}"



df = pd.read_excel("/Users/haroldsmith/Desktop/CognatePredictor/conjoined_hash_table.xlsx")
df["Phonological Cognate"] = df["Phonological Cognate"].fillna(value = -1)
df = df.fillna("Blank").sort_values(by = ["Phonological Cognate"], ascending = False)

vals = np.arange(1,20,1)# range of cognate class, 1 - 19
df_no_label = df.loc[~df["Phonological Cognate"].isin(vals)]# locating unlabeled word
located = df.loc[df["Phonological Cognate"].isin(vals)] # locating labeled words
located = located.sample(located.shape[0])
located["Phonological Cognate"] = located["Phonological Cognate"].to_numpy() - 1


words_to_translate = df["tech_rep"].to_numpy()
associated_labels = df["Phonological Cognate"]
images = located["tech_rep"].to_numpy()
labels = located["Phonological Cognate"].to_numpy().astype(np.int32)

training_images = encoding(images)
testing_images = encoding(words_to_translate)

pairTrain, labelTrain = make_pairs(training_images,labels)
word_pairs, cognate = make_pairs(images, labels)
## FOr some reason the encoding is not the right shape
#print(training_images.shape)

def build_siamese_model(hp):
    # specify the inputs for the feature extractor network
    print()
    print("**********************")
    print("[INFO]TUNING MODEL...")

    inputs = Input(IMG_SHAPE)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(hp.Int(
        'units 1',
        min_value=32,
        max_value=512,
        step=16,
        default=128), (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(hp.Int(
        'units 2',
        min_value=32,
        max_value=512,
        step=32,
        default=128), (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))(x)
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(hp.Float(
                'Embeding Dim',
                min_value=12,
                max_value=60,
                default=48,
                step=12,
            ))(pooledOutput)
    # build the model
    featureExtractor = Model(inputs, outputs)
    
    imgA = Input(shape = IMG_SHAPE)
    imgB = Input(shape = IMG_SHAPE)
    
    #print(featureExtractor)
    
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    
    distance = Lambda(euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation= "sigmoid")(distance)
    model1 = Model(inputs=[imgA, imgB], outputs=outputs)
    #print(model1.summary())
    opt = tf.keras.optimizers.Adam(learning_rate=hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                ))
    
    print("[INFO] compiling model...")
    model1.compile(loss="binary_crossentropy", optimizer= opt,
	metrics=["accuracy"])
    return model1
    # return the model to the calling function
   

def tuning():
    tuner = RandomSearch(build_siamese_model,
                objective = 'val_accuracy',
                max_trials = 10,
                executions_per_trial = 1,
                directory = LOG_DIR)
                
    tuner.search(x = [pairTrain[:, 0], pairTrain[:, 1]],
            y = labelTrain[:],
            epochs = 50,
            batch_size = 64,
            validation_split=(0.3) 
            
        )
    return tuner
        
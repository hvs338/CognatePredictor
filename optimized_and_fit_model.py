from build_and_tune_model import *


MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])


tuner = tuning()
tuned_param = tuner.get_best_hyperparameters()[0].values

class myCallback(Callback):
    def on_epoch_end(self,epoch,logs={}):
        if((logs.get('accuracy') > 0.9) and (logs.get('val_accuracy') > 0.74)):
            print("\n[INFO] Reached target accuracy so canceling training")
            self.model.stop_training = True

def build_siamese_model():
    # specify the inputs for the feature extractor network
    inputs = Input(IMG_SHAPE)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(tuned_param['units 1'], (2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(tuned_param['dropout_1'])(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(tuned_param['units 2'], (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(tuned_param['dropout_2'])(x)
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(tuned_param['Embeding Dim'])(pooledOutput)
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
    opt = tf.keras.optimizers.Adam(learning_rate=tuned_param['learning_rate'])
    print()
    print("**********************")
    print("[INFO] COMPILING MODEL...")
    print("**********************")
    print()
    model1.compile(loss="binary_crossentropy", optimizer= opt,
	metrics=["accuracy"])
    return model1

model = build_siamese_model()
callbacks = myCallback()

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
    
	plt.show()

# fittins the 
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_split=(0.2),
    batch_size= BATCH_SIZE, 
    epochs= 100,
    callbacks = callbacks)
    # serialize the model to disk<br>
print()
print("**********************")
print("[INFO] SAVING SIAMESE MODEL...")
print("**********************")
print()
model.save(MODEL_PATH)
# plot the training history<br>
print()
print("**********************")
print("[INFO] PlOTTING TRAINING DATA...")
print("**********************")
print()
plot_training(history, PLOT_PATH)
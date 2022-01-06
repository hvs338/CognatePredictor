# CognatePredictor
This is a Siamese Convolutional Nueral Network Trained to detect cognates between words in Meso-American Languages
It works by first encoding a given technical representation of a word as a psuedo-image of its phonemes. Each character is encoding as a one hot vector
of the sounds associated with a certain character. Together all the characters in the word make up the image.

In order to run and train the netwok all that one must do is run the jupyter notebook from start to finish. All of the accompany python files are helper files.
The model is built entirely on Keras Tensorflow, and tuned using the hyperparameter tuning module found in tensorflow. The ability to use a prection tool is
still being worked on, and will hopefully be implemented in the future.


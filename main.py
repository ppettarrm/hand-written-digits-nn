import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf

mnist = tf.keras.datasets.mnist;
[xTrain, yTrain], [xTest, yTest] = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis=1);
xTest = tf.keras.utils.normalize(xTest, axis=1);


model = tf.keras.models.load_model('HandWrittenModel.h5');

number = 0;
while os.path.isfile(f"numbers/{number}.png"):
    try:
        img = cv2.imread(f"numbers/{number}.png")[:,:,0];
        img = np.invert(np.array([img]));
        prediction = model.predict(img);
        print(f"Number {number}");
        print(f"This digit is probably a {np.argmax(prediction)}!");
        plt.imshow(img[0], cmap=plt.cm.binary);
        plt.show();
    except:
        print("Error");
    finally:
        number = number + 1;
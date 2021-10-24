import os
import cv2
# import tensorflow
import tensorflow as tf

# helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pick up the test_images from the relevant directory


def createModel():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    assert train_images.shape == (60000, 28, 28)
    assert test_images.shape == (10000, 28, 28)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images/255.0
    test_images = test_images/255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])

    #model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs = 10)

    trainModelMetrics = model.evaluate(train_images, train_labels, verbose = 1)

    testModelMetrics = model.evaluate(test_images, test_labels, verbose = 1)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    return probability_model

def importData(folder):
    currentDir = os.path.abspath(os.getcwdb())

    # currentDir returns bytes rather than a str so we join the paths as bytes objects and decode them into strings
    folder = str.encode(folder)
    destinationDir = os.path.join(currentDir, folder).decode("utf-8")

    files = [os.path.join(destinationDir, fileName) for fileName in os.listdir(destinationDir)]

    images = [cv2.imread(file) for file in files]
    images = [cv2.resize(img, (28, 28)) for img in images]

    img = images[0]

    gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, 2)/255

    return np.expand_dims(gray, 0)

def makePredictions(model, inputFiles):

    probability_model = model
    
    predictions = probability_model.predict(inputFiles)
    print(predictions)

def main():
    dataForPredictions = importData('predictions')
    #print(f"Shape of predictions data: {dataForPredictions.shape}")

    model = createModel()
    makePredictions(model, dataForPredictions)

if __name__ == "__main__":
    main()


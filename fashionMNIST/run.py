import os
import cv2
# import tensorflow
import tensorflow as tf

# helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    model.fit(train_images, train_labels, epochs = 1)

    trainModelMetrics = model.evaluate(train_images, train_labels, verbose = 1)
    print(f"The training loss is {trainModelMetrics[0]} and the training acc is {trainModelMetrics[1]}")

    testModelMetrics = model.evaluate(test_images, test_labels, verbose = 1)
    print(f"The validation loss is {testModelMetrics[0]} and the validation acc is {testModelMetrics[1]}")

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    return probability_model

# nb some weirdness around image dimensionality - needs fixing
def importData(folder):
    currentDir = os.path.abspath(os.getcwdb())

    # currentDir returns bytes rather than a str so we join the paths as bytes objects and decode them into strings
    folder = str.encode(folder)
    destinationDir = os.path.join(currentDir, folder).decode("utf-8")

    files = [os.path.join(destinationDir, fileName) for fileName in os.listdir(destinationDir)]

    images = [cv2.imread(file) for file in files]

    # turn images to grayscale and resize
    images = [cv2.resize(img, (28,28)) for img in images]
    images = tf.image.rgb_to_grayscale(images)
    images = [np.expand_dims(img, 0) for img in images]

    return images
    
def makePredictions(model, inputFiles):

    class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
                5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

    probability_model = model

    counter = 0
    # iterate over imgs passed to function
    for file in inputFiles:
        predictions = probability_model.predict(file).tolist()[0]
        
        prediction = class_names[predictions.index(max(predictions))]

        print(f"We think that file # {counter} is a {prediction}")
        counter += 1

    
def main():
    
    dataForPredictions = importData('predictions')

    model = createModel()
    makePredictions(model, dataForPredictions)

if __name__ == "__main__":
    main()


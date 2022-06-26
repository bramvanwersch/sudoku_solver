from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as keras_backend
import matplotlib.pyplot as plt

import digits_from_image

keras_backend.set_image_data_format('channels_first')

# train a model for number identification

X_train = digits_from_image.get_training_images(9)
y_train = [8, 1, 9, 5, 8, 7, 1, 4, 9, 7, 6, 7, 1, 2, 5, 8, 6, 1, 7, 1, 5, 2, 9, 7, 4, 6, 8, 3, 9, 4, 3, 5, 8, 6, 3, 8,
           4, 2, 3, 2, 8, 6, 3, 7, 5, 2, 7, 5, 6, 1, 7, 9, 1, 4, 2, 9, 5, 6, 1, 3, 9, 4, 2, 9, 2, 7, 8, 9, 1, 2, 3, 7,
           6, 8, 3, 6, 1, 9, 5, 5, 6, 8, 7, 9, 4, 2, 1, 8, 7, 7, 4, 4, 5, 2, 3, 6, 4, 7, 7, 6, 9, 5, 8, 7, 2, 9, 3, 8,
           5, 4, 3, 1, 7, 5, 2, 3, 2, 8, 2, 3, 1, 6, 4, 7, 7, 6, 9, 5, 8, 7, 2, 9, 3, 8, 5, 4, 3, 1, 7, 5, 2, 3, 2, 8,
           2, 3, 1, 6, 4, 7, 7, 6, 9, 5, 8, 7, 2, 9, 3, 8, 5, 4, 3, 1, 7, 5, 2, 3, 2, 8, 2, 3, 1]
X_test = X_train
y_test = y_train

# Reshape to be samples*pixels*width*height
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One Hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), padding="same", activation='relu'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# second set of CONV => RELU => POOL layers
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# second set of FC => RELU layers
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# softmax classifier
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

#  - - - - - - - TEST single image - - - - - - - -

image = (X_test[1]).reshape(1, 1, 28, 28)  # 1->'2';
model_pred = (model.predict(image) > 0.5).astype("int32")
print('Prediction of model: {}'.format(model_pred[0]))

# - - - - - - TESTING multiple image - - - - - - - - - -

test_images = X_test[1:5]
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print("Test images shape: {}".format(test_images.shape))

for i, test_image in enumerate(test_images, start=1):
    org_image = test_image
    test_image = test_image.reshape(1, 1, 28, 28)
    prediction = (model.predict(test_image) > 0.5).astype("int32")

    print("Predicted digit: {}".format(prediction[0]))
    plt.subplot(220 + i)
    plt.axis('off')
    plt.title("Predicted digit: {}".format(prediction[0]))
    plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()

# - - - - - - - SAVE THE MODEL - - - - - - - -

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

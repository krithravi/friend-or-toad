from PIL import Image
import numpy as np
import os, cv2
# some handfulls of happy arrays!

data = []
labels = []

cane = os.listdir("caneToad")
for caneToad in cane:
    imag = cv2.imread("caneToad/"+caneToad)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

regFrogs = os.listdir("nonCaneToad")
for regFrog in regFrogs:
    imag = cv2.imread("nonCaneToad/"+regFrog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

# making some numpy arrays?
froggos = np.array(data)
labels = np.array(labels)

np.save("froggos", froggos)
np.save("labels", labels)

# optional, apparently only for juypyter?
froggos = np.load("froggos.npy")
labels = np.load("labels.npy")

s = np.arange(froggos.shape[0])
np.random.shuffle(s)
froggos = froggos[s]
labels = labels[s]

num_classes = len(np.unique(labels))
data_length = len(froggos)

# splitting into train and test - doing a 90, 10 split
(x_train,x_test)=froggos[(int)(0.1*data_length):],froggos[:(int)(0.1*data_length)]

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_length=len(x_train)
test_length=len(x_test)

# divvy up between test and train
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

# "one hot encoding?
import keras
from keras.utils import np_utils#One hot encoding

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

# done with data prep! *hopefully*

# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout#make model

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax")) # we only have 2 categories

model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the boi
model.fit(x_train,y_train,batch_size=50,epochs=100,verbose=1)

# test the thing
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

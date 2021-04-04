# imports
from PIL import Image
import numpy as np
import os, cv2, sys, getopt
from keras.initializers import glorot_uniform
import tensorflow as tf

#Reading the model from JSON file
with open('./model.json', 'r') as json_file:
    json_savedModel= json_file.read()

#load the model architecture
model = tf.keras.models.model_from_json(json_savedModel)
#model.summary()

# predict on a single img
def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_animal_name(label):
    if label == 0:
        return "cane toad"
    if label == 1:
        return "regular frog"

def predict_animal(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    label = 1

    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a,verbose=1)
    print(score)

    label_index = np.argmax(score)
    print(label_index)
    acc = np.max(score)
    animal = get_animal_name(label_index)
    res = "This frog is a " + animal + " with accuracy = " + str(acc)
    print(res)
    return res


predict_animal(sys.argv[1])

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os, cv2, sys, getopt
from keras.initializers import glorot_uniform
import tensorflow as tf

app = Flask(__name__)

#Reading the model from JSON file
with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()

#load the model architecture
model = tf.keras.models.model_from_json(json_savedModel)

def convert_to_array(img):
    #img = cv2.imread(img)
    img = Image.fromarray(img, 'RGB')
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

    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a,verbose=1)

    label_index = np.argmax(score)
    acc = np.max(score)
    animal = get_animal_name(label_index)
    res = "This frog is a " + animal + " with accuracy = " + str(acc)
    #print(res)
    return res

@app.route('/', methods=['GET', 'POST'])
def toad():
    if request.method == "POST":
        #image_data = request.files("pic")
        image_data = request.files["pic"]
        npimg = np.fromfile(image_data, np.uint8)
        image_data = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return "<h1>Friend or Toad?</h1> <br>" + str(predict_animal(image_data))
    else:
        return "oh no! no data!"

################## DEALING WITH HTMLs AND STUFF ###########################################
@app.route('/form', methods=['GET', 'POST'])
def form():
   return render_template('form.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

    image_data = request.files["pic"]
    npimg = np.fromfile(image_data, np.uint8)
    image_data = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return "<h1>Friend or Toad?</h1> <br>" + str(predict_animal(image_data))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


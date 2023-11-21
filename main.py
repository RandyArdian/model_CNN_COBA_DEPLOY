from flask import Flask,jsonify,request
from tensorflow import keras
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import uuid
from io import BytesIO
import os

app = Flask(__name__)

# Load Model dengan flask
label = ['Tumor glioma',  
       'Tumor meningioma',
       'Tumor notumor', 
       'Tumor pituitary']

model = keras.models.load_model('randy-BrainTumorModel-94.22.h5')

def prediksiii(gambar): 
    Loaded=np.asarray(gambar)/255.0
    Loaded = Loaded.reshape(1, 224,224,3)
    preddd= model.predict(Loaded)
    prediksi= label[np.argmax(preddd)]
    return prediksi

@app.route('/PrediksiCNN', methods=['POST'])
def prediksiApi():
    if request.method == 'POST':
        data = request.get_json()
        if data['password'] == "zxcvbnm":
               # Mendekode base64
               img_data = data['data']
               img_binary = base64.b64decode(img_data)
               image = Image.open(BytesIO(img_binary))
               image = image.convert("RGB")
		       
               image= image.resize((224,224))
		       # Panggil method prediksi CNN
               Hasilprediksi = prediksiii(image)
		       # Kirim hasil prediksi dalam bentuk JSON
               return jsonify(
			    {
				'data': 'success',
	            'Klasifikasi':Hasilprediksi
			     }
			      ),200
        else:
              return jsonify(
			    {
				  'Failed':"Password is false"
			     }
			      ),404
@app.route('/')
def rera():
       return jsonify(
			{
			'Api':"Ok"
		  }
		),200              
if __name__ =='__main__':
	#app.debug = True
	app.run()
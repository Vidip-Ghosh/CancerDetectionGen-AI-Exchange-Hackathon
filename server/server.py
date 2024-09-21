from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import os
import numpy as np
from flask_cors import CORS
import collections
collections.Iterable = collections.abc.Iterable
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
app = Flask(__name__)
CORS(app)

ALLCancerModel = tf.keras.models.load_model("./Models/ALLCancer.h5")
brainCancerModel = tf.keras.models.load_model("./Models/BrainCancer.h5")
breastCancerModel = tf.keras.models.load_model("./Models/breastCancer.h5")
cervicalCancerModel = tf.keras.models.load_model("./Models/CervicalCancer.h5")
kidneyCancerModel = tf.keras.models.load_model("./Models/kidneyCancer.h5")
lungCancerModel = tf.keras.models.load_model("./Models/lungModel.h5")
lymphomaModel = tf.keras.models.load_model("./Models/LymphCancer.h5")
oralCancerModel = tf.keras.models.load_model("./Models/OralCancer.h5")

ALLCancerClassName={0: 'all_benign', 1:'all_early' , 2:'all_pre', 3:'all_pro'}

cervicalCancerClassName={
    0:'cervix_dyk',
    1:'cervix_koc',
    2:'cervix_mep',
    3:'cervix_pab',
    4:'cervix_sfi'
}

oralCancerClassName={
    0: 'oral_normal',
    1: 'oral_scc'
}
brainCancerClassName={0: 'brain_glioma', 1:'brain_menin' , 2:'brain_tumor'}
lymphomaCancerClassName={0:'lymph_cll', 1:'lymph_fl', 2:'lymph_mcl'}
breastCancerClassname={0:'breast_benign', 1:'breast_malignant'}
kidneyCancerClassname={0:'kidney_normal', 1:'kidney_tumor'}
lungCancerClassName={0:'colon_aca', 1:'colon_bnt', 2:'lung_aca',3:'lung_bnt',4:'lung_scc'}

def loadImg(imgPath): 
    img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def home(): 
    print("Received a POST request")
    file = request.files['file']
    file_path = os.path.join("./", file.filename)
    file.save(file_path)
    
    cancer_type = request.form['cancer_type']
    print("Cancer type: ",cancer_type)
        
    img = loadImg(file_path)
    if cancer_type=='brain-cancer': 
        predictions = brainCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": brainCancerClassName[predicted_class[0]]})
    elif cancer_type=='acute-lymphoblastic-leukemia': 
        predictions = ALLCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": ALLCancerClassName[predicted_class[0]]})
    elif cancer_type=='cervical-cancer':
        predictions = cervicalCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": cervicalCancerClassName[predicted_class[0]]})
    elif cancer_type=='oral-cancer': 
        predictions = oralCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": oralCancerClassName[predicted_class[0]]})
    elif cancer_type=='lymphoma': 
        predictions = lymphomaModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": lymphomaCancerClassName[predicted_class[0]]})
    elif cancer_type=='kidney-cancer':
        predictions = kidneyCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": kidneyCancerClassname[predicted_class[0]]})
    elif cancer_type=='lung-and-colon-cancer': 
        predictions = lungCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": lungCancerClassName[predicted_class[0]]})
    elif cancer_type=='breast-cancer':
        predictions = breastCancerModel.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        print("Predicted class: ",predicted_class) 
        os.remove(file_path)
        return jsonify({"prediction": breastCancerClassname[predicted_class[0]]}) 
    else: 
        return jsonify({"message": "Please select a correct class category with respect to image."})

if __name__ == '__main__':
    app.run(debug=True)

import sys,os
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.losses
import time 
from loss_j3_input import loss_wiener_j3
from read_input import load_config

if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

for key, value in input_data.items():
    print(f'{key} = {value}')
    
path = os.path.dirname(__file__)
name_model = path + "/models/"+input_data["name_model"]
dataset = np.load(path + "/data/" + input_data["name_test"])

data_test = dataset[:,:,1:]
data_test = np.reshape(data_test, [1,512,512,3])
print(np.shape(data_test))

name_result = path + "/result/" + input_data["name_result_cnn"]

channels_out = 1
npad = int(input_data["npad"])
img_shape = (768,768,3)
deviation = float(input_data["deviation"])
map_rescale_factor = 1./deviation
batch_size = int(input_data["batch_size"])

tf.keras.losses.custom_loss = loss_wiener_j3

model=load_model(name_model,custom_objects={'loss_wiener_j3':loss_wiener_j3})

def periodic_padding(images,npad):
    if len(images.shape)==4:
        images = np.pad(images,pad_width=((0,0),(npad,npad),(npad,npad),(0,0)),mode='wrap')
    if len(images.shape)==3:
        images = np.pad(images,pad_width=((npad,npad),(npad,npad),(0,0)),mode='wrap')        
    return images

images_test = periodic_padding(data_test, npad)

images_test[...,0] *= map_rescale_factor


t0 = time.time()
result = model.predict(images_test)
t1 = time.time()
    
result = result/map_rescale_factor
    
total = t1-t0 
print('prediction time', total)
np.save(name_result, result)

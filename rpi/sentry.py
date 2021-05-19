#this is a raspberry pi with a camera and a speaker
#it beeps if it sees a Starling or Squirrel
#it sends you an email attachment of bluebirds

import time
from picamera import PiCamera
import numpy as np
import tflite_runtime.interpreter as interpreter_wrapper
import io
from PIL import Image
import pygame
import datetime
from matplotlib import pyplot
import threading
from bluebird_email import send_bluebird_email

classes = ['background','starling','bluebird','squirrel','finch','robin','titmouse','dove','sparrow','food']

def preprocess_image(img_arr, model_image_size):
    image_data = np.array(img_arr.resize(model_image_size, 0))
    image_data = image_data/255
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data.astype('float32')
    return image_data

pygame.init()
pygame.mixer.init()
files = ['beep-01.mp3','beep-02.mp3','beep-01.mp3','beep-09.mp3']
pygame.mixer.music.load(files[0])
def beeping():
    file = np.random.choice(files)
    pygame.mixer.music.load(file)
    while True:
        pygame.mixer.music.play()
        global stop_beeping
        if stop_beeping:
            break
stop_beeping = True

model_path = 'seg.tflite' #yolov3_tiny_birds.tflite'
interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
camera = PiCamera()
while(True):
    start = time.time()
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    image = Image.open(stream)
    stream.flush()
    #image = Image.open('birds/sparrow.jpg')
    
    image_data = preprocess_image(image, (height, width))
    interpreter.set_tensor(input_details[0]['index'], image_data.astype('float32'))
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predictions = np.argmax(predictions, axis=3)
    if np.count_nonzero(predictions == 1) > 500:
        pygame.mixer.music.play(0)
        time.sleep(1)
        file = np.random.choice(files)
        pygame.mixer.music.load(file)
        print('starling spotted')
        image.save('./birds/starling.jpg')
    if np.count_nonzero(predictions == 3) > 1000:
        if stop_beeping == True:
            stop_beeping = False
            t1 = threading.Thread(target=beeping)
            t1.start()
        print('squirrel spotted')
        image.save('./birds/squirrel.jpg')
    else:
        if stop_beeping == False:
            stop_beeping = True
            t1.join()
    if np.count_nonzero(predictions == 2) > 1000:
        print('bluebird spotted')
        image.save('./birds/bluebird.jpg')
        #send_bluebird_email()
    #image.save('./birds/temp.jpg')
    for i in range(10):
        print(classes[i],'=', np.count_nonzero(predictions == i))
    if False:
        pyplot.subplot(1,2,1)
        pyplot.imshow(predictions[0])
        pyplot.subplot(1,2,2)
        pyplot.imshow(image_data[0])
        pyplot.show()
    print("total time: {:.2f}s".format((time.time() - start)))

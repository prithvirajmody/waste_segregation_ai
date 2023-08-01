from tkinter import *
import customtkinter
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoCapture, imwrite

#Image prediction through model

def pred(imgPath):
    model = load_model(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\waste_segregation_model_v2.h5')
    img = cv2.imread(imgPath)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    resize = tf.image.resize(img, (256, 256))
    plt.imshow(cv2.cvtColor(resize.numpy(), cv2.COLOR_BGR2RGB))

    yhat = model.predict(np.expand_dims(resize/255, 0))
    
    if yhat >= 0.5:
        resultLabel = customtkinter.CTkLabel(root, width=20, height=20, fg_color='transparent', text="Recyclable")
    else:
        resultLabel = customtkinter.CTkLabel(root, width=20, height=20, fg_color='transparent', text="Organic")

    resultLabel.pack(padx=10, pady=5)

#Image capturing code
def image_cap():
    cam = VideoCapture(0)
    result, img = cam.read()

    if result:
        impath = r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\UserInput.png'
        imwrite('UserInput.png', img)
        add_input_image = ImageTk.PhotoImage(Image.open(impath).resize((250, 250), Image.ANTIALIAS))
        inputImage = customtkinter.CTkLabel(root, image=add_input_image, text=None)
        inputImage.pack(padx=10)

        pred(imgPath=impath)
  
    else:
        print("No image detected. Please! try again")


#Create dark mode version
customtkinter.set_appearance_mode('Dark')
customtkinter.set_default_color_theme('dark-blue')

#Tkinter Boilerplate
root = customtkinter.CTk()
root.title('Waste Segregator')
#root.iconbitmap(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\assets\camera_button.png')
root.geometry('300x400')

title_label = customtkinter.CTkLabel(root, text="Check waste type", fg_color='transparent', height=25, width=25, font=('arial', 24))
title_label.pack(padx=20,pady=10)

title_underliner = customtkinter.CTkLabel(root, height=8, width=300, fg_color='white', text=None)
title_underliner.pack()


#Define Images
add_camera_image = ImageTk.PhotoImage(Image.open(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\assets\camera_button.png').resize((150, 150), Image.ANTIALIAS))

#Creating Buttons
camera_access_button = customtkinter.CTkButton(master=root, image=add_camera_image, text=None, height=13, width=13, command=image_cap)
camera_access_button.pack(pady=20, padx=20)

root.mainloop()
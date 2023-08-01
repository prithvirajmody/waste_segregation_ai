from tkinter import *
import customtkinter
from PIL import Image, ImageTk

#Create dark mode version
customtkinter.set_appearance_mode('Dark')
customtkinter.set_default_color_theme('dark-blue')

#Predict button function
def func_predict_button():
    root.destroy()
    exec(open(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\GUI_PredictPage.py').read())

def func_info_button():
    root.destroy()
    


#Tkinter Boilerplate
root = customtkinter.CTk()
root.title('Waste Segregator')
#root.iconbitmap(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\assets\camera_button.png')
root.geometry('300x400')

title_label = customtkinter.CTkLabel(root, text="Waste Segregation App", fg_color='transparent', height=25, width=25, font=('arial', 24))
title_label.pack(padx=20,pady=10)

title_underliner = customtkinter.CTkLabel(root, height=8, width=300, fg_color='white', text=None)
title_underliner.pack()

#Option button 1 - Predict
predict_button = customtkinter.CTkButton(master=root, text="Predict", command=func_predict_button)
predict_button.pack(pady=40)

#Option button 2 - Information
info_button = customtkinter.CTkButton(master=root, text="Additional Information")
info_button.pack(pady=40)

root.mainloop()
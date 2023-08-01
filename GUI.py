from tkinter import *
import customtkinter
from PIL import Image, ImageTk

#Create dark mode version
customtkinter.set_appearance_mode('Dark')
customtkinter.set_default_color_theme('dark-blue')

#Tkinter Boilerplate
root = customtkinter.CTk()
root.title('Waste Segregator')
#root.iconbitmap(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\assets\camera_button.png')
root.geometry('500x500')

#Define Images
add_camera_image = ImageTk.PhotoImage(Image.open(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\assets\camera_button.png').resize((100, 100), Image.ANTIALIAS))

#Creating Buttons
camera_access_button = customtkinter.CTkButton(master=root, image=add_camera_image, text=None)
camera_access_button.pack(pady=20, padx=20)

root.mainloop()
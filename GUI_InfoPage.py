from tkinter import *
import customtkinter
from cv2 import *
from PIL import Image, ImageTk

#Create dark mode version
customtkinter.set_appearance_mode('Dark')
customtkinter.set_default_color_theme('dark-blue')

#Image capturing code
def image_cap():
    cam = VideoCapture(0)
    result, img = cam.read()

    if result:
        imshow("Camera Input", img)
        imwrite("UserInput.png", img)
  
        waitKey(0)
        destroyWindow("Camera Input")
  
    else:
        print("No image detected. Please! try again")


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
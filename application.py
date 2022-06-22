from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import threading
import webbrowser
from tkinter import messagebox
from calibrate import calibrate
import main

# Init
tk = Tk()
windowWidth = tk.winfo_reqwidth()
windowHeight = tk.winfo_reqheight()
positionRight = int(tk.winfo_screenwidth() / 3 - windowWidth / 3)
positionDown = int(tk.winfo_screenheight() / 3 - windowHeight / 1)
tk.geometry(f"800x510+{positionRight}+{positionDown}")
tk.resizable(width=False, height=False)

tk.title("TU Digital Electrochemistry Lab")
F1 = Frame(tk)
F2 = None
F3 = None
F1.grid(row=0, column=0, pady=25, padx=25)
l1 = Label(F1, text="Original Image", font="bold")
l1.grid(row=0, column=0)
L1 = Label(F1, text="Original", height="25", width="52", bd=0.5, relief="solid")
L1.grid(row=1, column=0, pady=10, padx=15)
l2 = Label(F1, text="Modified Image", font="bold")
l2.grid(row=0, column=1)
L2 = Label(F1, text="Modified", height="25", width="52", bd=0.5, relief="solid")
L2.grid(row=1, column=1)
global_return = 0


# Image Select and Save
def Image_Select():
    global hsv
    global tkimage
    global original
    global img_rgb
    global imageselect

    imageselect = filedialog.askopenfilename(initialdir="Desktop",
                                             filetypes=[('Image files', '*.png'), ('Image files', '*.jpg')])

    if not imageselect:
        return
    print(imageselect)

    try:
        original = cv2.imread(imageselect)

        # Resimi Aç
        im = Image.open(imageselect)
        im.thumbnail((360, 360))
        tkimage = ImageTk.PhotoImage(im)

        img_rgb = np.array(im)


        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Widgetler
        L1 = Label(F1, image=None)  # , image = tkimage)
        L1.config(image=tkimage)

        L2 = Label(F1, image=None)
        L2.config(image=tkimage)

        L1.grid(row=1, column=0)
        L2.grid(row=1, column=1)
        saveBTN.config(state="normal", cursor="hand2")
        tk.geometry("1200x510")
    except:
        return



v1 = DoubleVar()
v2 = DoubleVar()
v3 = DoubleVar()
v4 = DoubleVar()
v5 = DoubleVar()
v6 = DoubleVar()
v7 = DoubleVar()
v8 = DoubleVar()
v9 = DoubleVar()
v10 = DoubleVar()
v11 = DoubleVar()


def hsv_buttons():
    global l_h
    global l_s
    global l_v
    global u_h
    global u_s
    global u_v
    global F2
    global LabelHSV
    if F3:
        F3.grid_forget()
        F3.destroy()
    F2 = Frame(tk)
    F2.place(x=900, y=50)

    l_h_lbl = Label(F2, text="Calibration Complete.")
    l_h_lbl.grid(row=0, column=0, sticky=W, padx="100")


def analyze_buttons():
    pass


def Sıfırla():
    LabelHSV["text"] = ""
    L2 = Label(F1, image=tkimage)
    L2.image = tkimage
    L2.grid(row=1, column=1)
    l_h.set(0)
    l_s.set(0)
    l_v.set(0)
    u_h.set(0)
    u_s.set(0)
    u_v.set(0)


def write_file():
    global global_return

    filename = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("PNG file", "*.png")])
    if not filename:
        return
    cv2.imwrite(f"{filename}.png", global_return)
    label = Label(F1, text="Saved.", font="bold")
    label.grid(row=2, column=1, pady=27)
    label.after(2000, label.destroy)


def calibrate_img(*args):
    global global_return

    calibrated_img = calibrate(original, 2)
    global_return = calibrated_img
    im = Image.fromarray(calibrated_img)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")


def analyze_img(*args):
    global global_return
    global option_variable

    if option_variable.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    img = main.main(original, option_variable.get())
    global_return = img
    im = Image.fromarray(img)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")



def calibrate_button():
    threading.Thread(target=calibrate_img).start()
    threading.Thread(target=hsv_buttons).start()


def analysis_button():
    threading.Thread(target=analyze_img).start()
    threading.Thread(target=analyze_buttons).start()


def trgt2():
    threading.Thread(target=Image_Select).start()


def trgt3():
    threading.Thread(target=write_file).start()

# Open Button
B1 = Button(tk, text="Open Image", command=trgt2)
B1.config(cursor="hand2")
B1.place(x=180, y=450)

# Save as Button
saveBTN = Button(tk, text="Save As", command=trgt3)
saveBTN.config(state="disabled")
saveBTN.place(x=565, y=450)


# Calibrate Button
hsv_btn = Button(tk, text="Calibrate", width=13, command=calibrate_button)
hsv_btn.bind("<ButtonRelease-1>", calibrate_img)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=80)

# Analysis Button
hsv_btn = Button(tk, text="Analyze", width=13, command=analysis_button)
hsv_btn.bind("<ButtonRelease-1>", analyze_img)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=115)

# Choice button
choices = ['PbO2', 'PbI2', 'PEDOT']
option_variable = StringVar(tk)
option_variable.set('Select Type')
w = OptionMenu(tk, option_variable, *choices)
w.place(x=800, y=35)


def callback(url):
    webbrowser.open_new(url)

tk.mainloop()

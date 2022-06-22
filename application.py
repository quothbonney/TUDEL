#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Başlangıç Tarihi: 06.05.2021 - 17:06
# Bitiş Tarihi: 11.05.2021 - 21:00
# instagram.com/yazilimfuryasi
# yazilimfuryasi.com

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import threading
import webbrowser
from calibrate import calibrate

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

    l_h_lbl = Label(F2, text="Lower - H | Ton")
    l_h_lbl.grid(row=0, column=0, sticky=W, padx="100")



def blrr():
    global orjn

    saveBTN.config(state="disabled", cursor="")

    bilFilter = cv2.bilateralFilter(img_rgb, blur.get(), blur2.get(), blur3.get())
    orjn = cv2.bilateralFilter(original, blur.get(), blur2.get(), blur3.get())

    imgtk3 = ImageTk.PhotoImage(image=Image.fromarray(bilFilter))

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3
    L2.grid(row=1, column=1)

    saveBTN.config(state="normal", cursor="hand2")


def blurring():
    global F3
    global blur
    global blur2
    global blur3
    if F2:
        F2.grid_forget()
        F2.destroy()

    def trgt_scale():
        threading.Thread(target=blrr).start()

    F4 = Frame(tk)
    F4.place(x=900, y=50)

    blr_lbl2 = Label(F4, text="Pixel")
    blr_lbl2.grid(row=0, column=0, sticky=W, padx="100")

    blur1 = Scale(F4, length=255, variable=v7, from_=1, to=100, troughcolor="red", orient=HORIZONTAL)
    blur1.bind("<ButtonRelease-1>", trgt_scale)
    blur1.grid(row=1, column=0, padx=27)

    blr_lbl3 = Label(F4, text="Renk")
    blr_lbl3.grid(row=2, column=0, sticky=W, padx="100")

    blur2 = Scale(F4, length=255, variable=v8, from_=1, to=100, troughcolor="white", orient=HORIZONTAL)
    blur2.bind("<ButtonRelease-1>", trgt_scale)
    blur2.grid(row=3, column=0, padx=27)


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


def Kayıt():
    global global_return
    # Yeni Resmi Kayıt Et
    dosyaAdi = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("PNG file", "*.png")])
    if not dosyaAdi:
        return
    cv2.imwrite(f"{dosyaAdi}.png", global_return)
    KayıtMesajı = Label(F1, text="Kayıt Edildi.", font="bold")
    KayıtMesajı.grid(row=2, column=1, pady=27)
    KayıtMesajı.after(2000, KayıtMesajı.destroy)


def calibrate_img(*args):
    global global_return
    calibrated_img = calibrate(original, 5)
    global_return = calibrated_img
    im = Image.fromarray(calibrated_img)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")




def maske_trgt():
    threading.Thread(target=calibrate_img).start()
    threading.Thread(target=hsv_buttons).start()


def blur_trgt2():
    threading.Thread(target=blurring).start()


def trgt2():
    threading.Thread(target=Image_Select).start()


def trgt3():
    threading.Thread(target=Kayıt).start()


B1 = Button(tk, text="Open Image", command=trgt2)
B1.config(cursor="hand2")
B1.place(x=180, y=450)

hsv_btn = Button(tk, text="HSV Maskeleme", width=13, command=maske_trgt)
hsv_btn.bind("<ButtonRelease-1>", calibrate_img)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=56)

saveBTN = Button(tk, text="Save As", command=trgt3)
saveBTN.config(state="disabled")
saveBTN.place(x=565, y=450)


def callback(url):
    webbrowser.open_new(url)

tk.mainloop()

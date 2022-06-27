from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import threading
import webbrowser
from tkinter import messagebox
from features.calibrate import calibrate
from features import dimensions, selection, analysis
from features.mask import Mask

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


def update_image(dst):
    global_return = dst
    im = Image.fromarray(dst)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")

# Image Select and Save
def Image_Select():
    global hsv
    global tkimage
    global original
    global img_rgb
    global imageselect
    global global_return

    imageselect = filedialog.askopenfilename(initialdir="Desktop",
                                             filetypes=[('Image files', '*.png'), ('Image files', '*.jpg')])

    if not imageselect:
        return
    print(imageselect)

    try:
        original = cv2.imread(imageselect)
        global_return = original

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
    l_h_lbl.grid(row=0, column=0, sticky=W, padx="30")


def analyze_buttons():
    global global_return
    global option_variable

    if option_variable.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    mask = Mask(global_return, option_variable.get())
    original_mask = mask.deposition_mask()
    error_mask = analysis.errors(original_mask, mask, global_return)




    cv2.imshow('errors2', error_mask)
    cv2.imshow('original2', original_mask)
    ratio: str = analysis.percent_imp(error_mask, original_mask, global_return)

    F2 = Frame(tk)
    F2.place(x=900, y=70)
    lbl = Label(F2, text=f"Percent Imperfection: {ratio}")
    lbl.grid(row=0, column=0, sticky=W, padx="30")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_file():
    global global_return
    ret = cv2.cvtColor(global_return, cv2.COLOR_BGR2RGB)

    filename = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("PNG file", "*.png")])
    if not filename:
        return
    cv2.imwrite(f"{filename}.png", ret)
    label = Label(F1, text="Saved.", font="bold")
    label.grid(row=2, column=1, pady=27)
    label.after(2000, label.destroy)


def calibrate_img(*args):
    global global_return

    global_return = calibrate(global_return, 2)

    im = Image.fromarray(global_return)
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

    # img = main.main(original, option_variable.get())

    mask = Mask(global_return, option_variable.get())
    error_mask = analysis.errors(mask, global_return)
    dst = analysis.show_errors(error_mask, global_return)


    global_return = dst
    im = Image.fromarray(dst)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")


def mask_img(*args):
    global global_return
    global option_variable

    mask = Mask(global_return, option_variable.get())
    original_mask = mask.deposition_mask()

    dep_masked = cv2.bitwise_and(global_return, global_return, mask=original_mask)
    update_image(dep_masked)

    return dep_masked


def dimension_img(*args):
    global option_variable

    if option_variable.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    width = dimensions.size(original, option_variable.get())
    F2 = Frame(tk)
    F2.place(x=900, y=90)

    a = Label(F2, text=f"Width: {width}px")
    a.grid(row=0, column=0, sticky=W, padx="30")

    new = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    height = dimensions.size(new, option_variable.get())
    F3 = Frame(tk)
    F3.place(x=900, y=110)
    b = Label(F3, text=f"Height: {height}px")
    b.grid(row=0, column=0, sticky=W, padx="30")

    F4 = Frame(tk)
    F4.place(x=900, y=130)
    c = Label(F4, text=f"Surface Area: {str(round(float(height) * float(width), 4))}px^2")
    c.grid(row=0, column=0, sticky=W, padx="30")

    F5 = Frame(tk)
    F5.place(x=900, y=150)
    d= Label(F5, text=f"\n\nNOTE: Values are in pixels. To get value in mm,\n divide by number of pixels in 1mm by \naligning film with a calibration slide.")
    d.grid(row=0, column=0, sticky=W, padx="30")


def calibrate_button():
    threading.Thread(target=calibrate_img).start()
    threading.Thread(target=hsv_buttons).start()


def analysis_button():
    threading.Thread(target=analyze_img).start()
    threading.Thread(target=analyze_buttons).start()


def dimension_button():
    threading.Thread(target=dimension_img).start()


def mask_button():
    threading.Thread(target=mask_img).start()

def manual_mask_button():
    global global_return

    left, top, right, bottom = selection.main(global_return)

    # Ensure that it won't slice backwards
    if top > bottom:
        bottom, top = top, bottom
    if left > right:
        right, left = left, right

    # Numpy slicing crop that I refuse to believe I'm smart enough to have thought of myself
    cropped = global_return[top:bottom, left:right]

    global_return = cropped
    update_image(global_return)

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

# Dimensions Button
hsv_btn = Button(tk, text="Dimensions", width=13, command=dimension_button)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=150)

# Auto Mask Button
hsv_btn = Button(tk, text="Auto Mask", width=13, command=mask_button)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=250)

# Manual Mask Button
hsv_btn = Button(tk, text="Manual Mask", width=13, command=manual_mask_button)
hsv_btn.config(cursor="hand2")
hsv_btn.place(x=800, y=285)

# Choice button
choices = ['PbO2', 'PbI2', 'PEDOT']
option_variable = StringVar(tk)
option_variable.set('Select Type')
w = OptionMenu(tk, option_variable, *choices)
w.place(x=800, y=35)


def callback(url):
    webbrowser.open_new(url)

tk.mainloop()

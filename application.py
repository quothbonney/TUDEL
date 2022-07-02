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
import json
import os
import sys

image_is_open = False
original = 0
# Init
tk = Tk()
tk.tk.call('tk', 'scaling', 2.0)


f = open("src/spectrum.json")
bound_map = json.load(f)

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

is_auto_masked = None
working_mask = [0]

def update_image(dst):
    global global_return, L2
    L2.config(image=None)
    L2.image = None
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

    filemenu.entryconfig("Save as", state="normal")
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
    global original

    if option_variable.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    mask = Mask(option_variable.get(), global_return)
    original_mask = mask.deposition_mask(global_return)
    error_mask = analyze_img()

    try:
        ratio: str = analysis.percent_imp(error_mask, original_mask, global_return)
    except ZeroDivisionError:
        ratio: str = "0.0000"
    F2 = Frame(tk)
    F2.place(x=900, y=70)
    lbl = Label(F2, text=f"Percent Imperfection: {ratio}")
    lbl.grid(row=0, column=0, sticky=W, padx="30")

    dst = analysis.show_errors(error_mask, global_return)
    update_image(dst)


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
    global is_auto_masked

    if option_variable.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')
        return

    if is_auto_masked == None:
        messagebox.showerror('TUDEL', 'Error: No mask selected.')
        return

    print(is_auto_masked)
    error_mask = analysis.errors(option_variable.get(), global_return, is_auto=is_auto_masked)

    return error_mask

    cv2.waitKey(0)


def mask_img(*args):
    global global_return
    global option_variable
    global working_mask

    mask = Mask(option_variable.get(), global_return)
    original_mask = mask.deposition_mask(global_return)
    working_mask = original_mask

    dep_masked = cv2.bitwise_and(global_return, global_return, mask=original_mask)
    update_image(dep_masked)
    global_return = dep_masked
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
    threading.Thread(target=analyze_buttons).start()


def dimension_button():
    threading.Thread(target=dimension_img).start()


def mask_button():
    global is_auto_masked
    is_auto_masked = True
    threading.Thread(target=mask_img).start()


def manual_mask_button():
    global global_return
    global is_auto_masked
    global working_mask
    is_auto_masked = False

    # Get the ratio between width and height in order to resize from size of one side
    shp: tuple = global_return.shape
    height_width_ratio = shp[0]/shp[1]
    size = 900
    dim = (size, int(size * height_width_ratio))

    # Get the points of the selected area
    resized = cv2.resize(global_return, dim, interpolation=cv2.INTER_AREA)
    left, top, right, bottom = selection.main(resized)

    rescaling_factor = shp[1]/size
    print(rescaling_factor)
    left = int(left * rescaling_factor)
    top = int(top * rescaling_factor)
    right = int(right * rescaling_factor)
    bottom = int(bottom * rescaling_factor)

    # Ensure that it won't slice backwards
    if top > bottom:
        bottom, top = top, bottom
    if left > right:
        right, left = left, right

    # Numpy slicing crop that I refuse to believe I'm smart enough to have thought of myself
    cropped = global_return[top:bottom, left:right]

    working_mask = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    global_return = cropped

    update_image(global_return)


def reset():
    global global_return, original
    global_return = original
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
choices = [choice for choice in bound_map]
option_variable = StringVar(tk)
option_variable.set('Select Type')
w = OptionMenu(tk, option_variable, *choices)
w.place(x=800, y=35)

# Reset button
reset_btn = Button(tk, text="Reset", width=9, command=reset)
reset_btn.config(cursor="hand2")
reset_btn.place(x=800, y=450)

#File Menu Only Buttons
def new_window():
    os.system("python3 ./application.py")


def donothing():
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()
   
menubar = Menu(tk)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New Window", command=new_window)
filemenu.add_command(label="Open", command=trgt2)
filemenu.add_command(label="Save as", command=trgt3)
filemenu.add_command(label="Close", command=tk.quit)

filemenu.add_separator()

# Save as is disabled by default to avoid error
filemenu.entryconfig("Save as", state="disabled")

filemenu.add_command(label="Exit", command=tk.quit)
menubar.add_cascade(label="File", menu=filemenu)
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo", command=donothing)

editmenu.add_separator()

editmenu.add_command(label="Cut", command=donothing)
editmenu.add_command(label="Copy", command=donothing)
editmenu.add_command(label="Paste", command=donothing)
editmenu.add_command(label="Delete", command=donothing)
editmenu.add_command(label="Select All", command=donothing)

menubar.add_cascade(label="Edit", menu=editmenu)
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=donothing)
menubar.add_cascade(label="Help", menu=helpmenu)

tk.config(menu=menubar)

tk.mainloop()
f.close()

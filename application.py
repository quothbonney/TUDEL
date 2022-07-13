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
from src.state import ApplicationState
import json
import os
import sys

# Init
tk = Tk()

state = ApplicationState()
state.original = 0

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
state.present = 0

is_auto_masked = None
working_mask = [0]

def update_image(dst):
    global L2
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    L2.config(image=None)
    L2.image = None
    state.present = dst
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
    global img_rgb
    global imageselect

    # Open menubars
    filemenu.entryconfig("Save as", state="normal")
    menubar.entryconfig("Edit", state="normal")

    imageselect = filedialog.askopenfilename(initialdir="Desktop",
                                             filetypes=[('Image files', '*.png'), ('Image files', '*.jpg')])

    if not imageselect:
        return
    print(imageselect)

    try:
        state.original = cv2.imread(imageselect)
        state.present = state.original

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
    global state

    if state.type.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    if state.is_masked is not True:
        messagebox.showerror('TUDEL', 'Error: Please select mask')

    mask = Mask(state.type.get(), state.present)
    state.original_mask = mask.deposition_mask(state.present)
    state.working_mask = analyze_img()

    try:
        ratio: str = analysis.percent_imp(state.working_mask, state.original_mask, state.present)
    except ZeroDivisionError:
        ratio: str = "0.0000"
    F2 = Frame(tk)
    F2.place(x=900, y=70)
    lbl = Label(F2, text=f"Percent Imperfection: {ratio}")
    lbl.grid(row=0, column=0, sticky=W, padx="30")

    dst = analysis.show_errors(state.working_mask, state.present)
    update_image(dst)


def write_file():
    global state
    ret = cv2.cvtColor(state.present, cv2.COLOR_BGR2RGB)

    filename = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("PNG file", "*.png")])
    if not filename:
        return
    cv2.imwrite(f"{filename}.png", ret)
    label = Label(F1, text="Saved.", font="bold")
    label.grid(row=2, column=1, pady=27)
    label.after(2000, label.destroy)


def calibrate_img(*args):
    global state

    if state.is_masked == True:
        messagebox.showerror('TUDEL', 'Error: Cannot Calibrate Mask')
        return
    
    if state.is_masked is True:
        messagebox.showerror('TUDEL', 'Error: Cannot calibrate mask')

    state.present = calibrate(state.present, 2)

    im = Image.fromarray(state.present)
    im.thumbnail((360, 360))
    imgtk3 = ImageTk.PhotoImage(image=im)

    L2 = Label(F1, image=imgtk3)
    L2.image = imgtk3

    L2.grid(row=1, column=1)
    saveBTN.config(state="normal", cursor="hand2")


def analyze_img(*args):
    global state

    if state.type.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')
        return

    if state.is_masked == False:
        messagebox.showerror('TUDEL', 'Error: No mask selected.')
        return

    state.working_mask = analysis.errors(state.type.get(), state.present, is_auto=state.is_auto_masked)

    return state.working_mask


def lines(*args):
    w = analysis.line_analysis(state.present)
    analysis.show_line_analysis(w)


def mask_img(*args):
    global state
    
    if state.type.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')
        return

    menubar.entryconfig("Analysis", state="normal")
    mask = Mask(state.type.get(), state.present)
    state.original_mask = mask.deposition_mask(state.present)

    dep_masked = cv2.bitwise_and(state.present, state.present, mask=state.original_mask)
    update_image(dep_masked)
    state.present = dep_masked
    return dep_masked


def dimension_img(*args):

    if state.type.get() == 'Select Type':
        messagebox.showerror('TUDEL', 'Error: Please select film type')

    if state.is_masked is not True:
        messagebox.showerror('TUDEL', 'No mask selected')


    width = dimensions.size(state.original, state.type.get())
    F2 = Frame(tk)
    F2.place(x=900, y=90)

    a = Label(F2, text=f"Width: {width}px")
    a.grid(row=0, column=0, sticky=W, padx="30")

    new = cv2.rotate(state.original, cv2.ROTATE_90_CLOCKWISE)
    height = dimensions.size(new, state.type.get())
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
    state.is_auto_masked = True
    state.is_masked = True
    threading.Thread(target=mask_img).start()
    

def manual_mask_button():
    global state 
    state.is_auto_masked = False
    state.is_masked = True

    # Get the ratio between width and height in order to resize from size of one side
    shp: tuple = state.present.shape
    height_width_ratio = shp[0]/shp[1]
    size = 900
    dim = (size, int(size * height_width_ratio))

    # Get the points of the selected area
    resized = cv2.resize(state.present, dim, interpolation=cv2.INTER_AREA)
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
    cropped = state.present[top:bottom, left:right]

    working_mask = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    state.present = cropped
    
    menubar.entryconfig("Analysis", state="normal")
    update_image(state.present)


    
def sat(chan):
    sats = analysis.saturation_histogram(channel=chan, image=state.present)
    analysis.show_saturations(sats)


def reset():
    state.is_auto_masked = False
    state.is_masked = False
    update_image(state.original)


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
state.type = StringVar(tk)
state.type.set('Select Type')
w = OptionMenu(tk, state.type, *choices)
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

# File bar
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New Window", command=new_window)
filemenu.add_command(label="Open", command=trgt2)
filemenu.add_command(label="Save as", command=trgt3)

filemenu.add_separator()

# Save as is disabled by default to avoid error
filemenu.entryconfig("Save as", state="disabled")

filemenu.add_command(label="Exit", command=tk.quit)
menubar.add_cascade(label="File", menu=filemenu)

# Edit bar
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Reset", command=reset)

editmenu.add_separator()

editmenu.add_command(label="Calibrate", command=calibrate_button)
editmenu.add_command(label="Dimension", command=dimension_button)

editmenu.add_separator()

editmenu.add_command(label="Auto Mask", command=mask_button)
editmenu.add_command(label="Manual Mask", command=manual_mask_button)

menubar.add_cascade(label="Edit", menu=editmenu)
menubar.entryconfig("Edit", state="disabled")

# Analysis bar
analysisbar = Menu(menubar, tearoff=0)

analysisbar.add_command(label="Imperfection", command=analysis_button)
analysisbar.add_command(label="Line by Line", command=lines)
analysisbar.add_command(label="Saturation", command=lambda: sat(1))
analysisbar.add_command(label="Value", command=lambda: sat(2))

menubar.add_cascade(label="Analysis", menu=analysisbar)
menubar.entryconfig("Analysis", state="disabled")


tk.config(menu=menubar)

tk.mainloop()
f.close()

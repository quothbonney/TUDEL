import numpy as np
import cv2
import matplotlib.pyplot as plt
from features.mask import Mask
from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


def mask_size(mask):
    # Count number of pixels in a mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # Set all pixels = 1, 0;
    pixels = cv2.countNonZero(thresh)  # Basically just a sum of all pixels
    return pixels


def saturation_histogram(image, hsvize=True):
    if hsvize is True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    channel = image[:,:,2]
    x = channel.flatten()

    fil = [p for p in x if p > 10]

    return fil


def errors(type, deposit, is_auto):
    mask = Mask(type, deposit)
    sobel = mask.sobel_mask()  # Get sobel mask
    edges = mask.edge_sobel_mask()
    if is_auto:
        final_mask = cv2.threshold(sobel-edges, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    else:
        final_mask = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        print("Manual Mask Confirmed")
    return final_mask


def show_errors(mask, image):
    green = np.zeros(image.shape, np.uint8)
    green[:] = (57, 255, 20)
    green_mask = cv2.bitwise_or(green, image, mask=mask)
    dst = cv2.addWeighted(green_mask, 0.5, image, 0.7, 0)

    return dst


def percent_imp(errors_mask, original_mask, image):
    result = image.copy()
    white= np.zeros(result.shape, np.uint8)
    white[:] = (255, 255, 255)
    errors_masked = cv2.bitwise_and(white, white, mask=errors_mask)
    dep_masked = cv2.bitwise_and(white, white, mask=original_mask)

    error_size = mask_size(errors_masked)
    deposit_size = mask_size(dep_masked)
    ratio = error_size / deposit_size

    ratio_string = "{0:.5f}%".format(ratio * 100)

    return ratio_string


def line_analysis(mask): 
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    
    widths = []
    for row in thresh:
        width = cv2.countNonZero(row)
        if width > 10:
            widths.append(width)
    
    return widths


def save(widths):
    txt_content = ""
    for w in widths:
        point = str(w)
        txt_content += point + "\n"

    fob=filedialog.asksaveasfile(filetypes=[('text file','*.txt')],
        defaultextension='.txt',initialdir='D:\\my_data\\my_html',
        mode='w')
    try:
        fob.write(txt_content)
        fob.close()
    except :
        print (" There is an error...")


def show_line_analysis(widths):
    window = Toplevel()
    window.title("Line by Line Analysis")
    window.geometry('%sx%s' % (600, 600))
    window.configure(background='grey')

    fig = Figure(figsize=(5,5), dpi=100) 
    plot1 = fig.add_subplot(111) # No I don't know why.
    plot1.plot(widths)
    plot1.set_xlabel('Row (px)')
    plot1.set_ylabel('Highlighted px')

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    btn = Button(window, text="Save Data", width=13, command=lambda: save(widths))
    btn.place(x=20, y=20)


    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()

    canvas.get_tk_widget().pack()

 

def show_saturations(sats):
    window = Toplevel()
    window.title("Saturation Histogram")
    window.geometry('%sx%s' % (600, 600))
    window.configure(background='grey')

    fig = Figure(figsize=(5,5), dpi=100) 
    plot1 = fig.add_subplot(111) # No I don't know why. 
  

    counts, bins = np.histogram(sats, bins=40, density=True)
    plot1.hist(bins[:-1], bins, weights=counts)
    plot1.set_xlabel('Saturation')
    plot1.set_ylabel('Probability')
   
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    btn = Button(window, text="Save Data", width=13, command=lambda: save(sats))
    btn.place(x=20, y=20)


    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()

    canvas.get_tk_widget().pack()






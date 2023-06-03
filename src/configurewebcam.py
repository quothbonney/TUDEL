import cv2
import tkinter as tk
import util
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


def webcam_window():
    def on_submit():
        selected_value = dropdown_var.get()
        print("Selected value:", selected_value)
        window.destroy()
        util.write_to_config(selected_value, "webcam_number")
        return selected_value

    # Create a Tkinter window
    window = tk.Toplevel()
    window.title("Line by Line Analysis")
    window.geometry('%sx%s' % (600, 600))
    window.configure(background='grey')

    fig, axs = plt.subplots(1, 5, figsize=(12, 4))

    # Create a variable to store the selected value from the dropdown
    dropdown_var = tk.StringVar()

    # Function to handle dropdown selection
    def on_dropdown_select(event):
        print("Selected value:", dropdown_var.get())

    captures = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        captures.append(frame)
        cap.release()
        cv2.destroyAllWindows()

    for i, array in enumerate(captures):
        if array is not None:
            axs[i].imshow(array, cmap='gray')
        else:
            axs[i].imshow(np.zeros((2, 2)), cmap='gray')

        axs[i].axis('off')

    # Display numbers 1 to 5 underneath each array
    for i, ax in enumerate(axs):
        ax.text(0.5, -0.2, str(i+1), transform=ax.transAxes, fontsize=12, horizontalalignment='center')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.1)

    # Show the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    # Create a dropdown menu
    dropdown = tk.OptionMenu(window, dropdown_var, *range(1, 6))
    dropdown.config(width=10)
    dropdown.pack()
    dropdown.place(x=20, y=20)

    # Create a submit button
    submit_btn = tk.Button(window, text="Submit", width=13, command=on_submit)
    submit_btn.pack()
    submit_btn.place(x=150, y=20)

    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()

    canvas.get_tk_widget().pack()

    window.mainloop()  # Add this line to start the Tkinter event loop
    
def get_webcam_number():
    result = webcam_window()
        


get_webcam_number()
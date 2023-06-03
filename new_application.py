import queue
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
from tkinter import scrolledtext
from src.selection import launch_select_window


class SingletonTextHandler:
    _instance = None
    # Holds messages in an array, indexes the last element of the array
    messages = [" "]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonTextHandler, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def add_message(cls, message):
        cls.messages.append(message)

f = open("data/spectrum.json")

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("TU Digital Electrochemistry Lab")
        self.geometry("1200x800")
        self.iconphoto(True, tk.PhotoImage(file='imgs/icon.png'))

        self.image_handler = ImageWranger()
        self.interface = Interface(self)

        self.resizable = False

class ImageWranger:
    def __init__(self):
        # Init to arbitrary zero matricies (avoids possible future errors)
        self.left = np.zeros((9, 9, 3))
        self.right = np.zeros((9, 9, 3))
        self.original = np.zeros((9, 9, 3))

    def select_image(self):
        imageselect = filedialog.askopenfilename(initialdir="Desktop",
                                             filetypes=[('Image files', '*.png'), ('Image files', '*.jpg'), ('Image files', '*.tif')])
        if not imageselect:
            return

        try:
            original = cv2.imread(imageselect)
            self.right = original
            self.left = original
        except:
            return

        SingletonTextHandler.add_message(f"Opened image {imageselect} successfully")

    def write_image(self):
        ret = cv2.cvtColor(self.right, cv2.COLOR_BGR2RGB)

        filename = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("PNG file", "*.png")])
        if not filename:
            return
        cv2.imwrite(f"{filename}.png", ret)
        SingletonTextHandler.add_message(f"Wrote image successfully to {filename}")

        return True

    def manual_mask(self):
        cropped = launch_select_window(self.left)
        SingletonTextHandler.add_message(f"Cropped image to size ({cropped.shape[0]}, {cropped.shape[1]})")
        self.right = cropped

class ImageView(tk.Label):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.image = None

    def set_image(self, image_array):
        """Loads and sets the image for this widget."""
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # Shrinks the image down to size for the boxes (only done on the frontend)
        img.thumbnail((360, 360))
        self.image = ImageTk.PhotoImage(img)
        self.config(image=self.image)

class ConsoleView(scrolledtext.ScrolledText):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(height="12", padx="12", pady="12", wrap=tk.WORD, state="disabled")
        self.q = SingletonTextHandler()

    def update(self):
        # Indexes the most recent message in the message buffer and writes it
        message = self.q.messages[-1]
        self.write_message(message)

        print(self.q.messages[-1])

    def write_message(self, message):
        # For some reason the ScrollingText will not update unless it is marked as being in the normal state
        # We are just rolling with it. a 2ms flicker is the least awful thing about this program
        self.configure(state="normal")
        self.insert(tk.END, message + '\n')
        self.configure(state="disabled")

    def clear_console(self):
        """Clears the console view."""
        self.delete('1.0', tk.END)

class ExperimentInterface(ttk.Frame):
    def __init__(self, master, state='disabled'):
        super().__init__(master)
        self.master: Interface = master
        self.grid(padx=25, pady=25)
        self.state = state

        self.create_widgets()

    def create_widgets(self):
        self.container = tk.Frame(self, width=200, height=500, relief=tk.RAISED, borderwidth=2)
        self.container.grid(row=0, column=0, columnspan=1, sticky=(tk.N, tk.S, tk.W, tk.E))

        dropdown_var = tk.StringVar()

        self.man_mask = ttk.Button(self.container, text="Manual Mask", state=self.state, command=lambda: (self.master.image_handler.manual_mask(), self.master.update_image_interface(2), self.master.console.update()))
        self.man_mask.grid(row=0, column=1, padx=10, pady=10)

    def change_state(self, state: bool):
       self.state = 'normal' if state else 'disabled'


class Interface(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.image_handler = master.image_handler
        self.grid(pady=25, padx=25)

        self.create_widgets()

    def create_widgets(self):
        # ------------- Image Boxes/Labels -------------
        # Labels for Left side
        self.label_left = ttk.Label(self, text="Original Image", font="bold")
        self.label_left.grid(row=0, column=0)
        # Bounding box of image
        self.label_view_left = tk.Label(self, text="Modified", height="19", width="52", bd=0.5, relief="solid")
        self.label_view_left.grid(row=1, column=0, pady=10, padx=10)
        # ImageView containing the original image
        self.view_left = ImageView(self, image=None)
        self.view_left.grid(row=1, column=0)

        # Same thing going on here
        self.label_right = ttk.Label(self, text="Modified Image", font="bold")
        self.label_right.grid(row=0, column=1)
        self.label_view_right = tk.Label(self, text="Modified", height="19", width="52", bd=0.5, relief="solid")
        self.label_view_right.grid(row=1, column=1, pady=10, padx=10)
        self.view_right = ImageView(self, text="Modified")
        self.view_right.grid(row=1, column=1)

        # -------------- Buttons ---------------
        self.open_button = ttk.Button(self, text="Open Image", command=lambda: (self.image_handler.select_image(), self.update_image_interface(2), self.console.update()))
        self.open_button.grid(row=2, column=0)

        self.save_button = ttk.Button(self, state="disabled", text="Save Image", command=lambda: (self.image_handler.write_image(), self.update_image_interface(2), self.console.update()))
        self.save_button.grid(row=2, column=1)

        self.divider = ttk.Separator(self, orient="vertical")
        self.divider.grid(row=2, column=0)

        self.experiments = ExperimentInterface(self)
        self.experiments.grid(column=2, row=0)

        # -------------- Console ---------------
        self.console = ConsoleView(self)
        self.console.grid(column=0, row=3, columnspan=2, sticky=(tk.N, tk.S, tk.W, tk.E))

    def update_image_interface(self, side: int):
        # Change button states depending on image loaded
        if self.image_handler.left is not None:
            self.save_button.config(state="normal")
        else:
            self.save_button.config(state="disabled")

        if side not in (0, 1, 2):
            raise Exception("Side must be (0, 1, 2)")

        if side == 1:
            imr = self.image_handler.left
            self.view_right.set_image(imr)
        elif side == 0:
            iml = self.image_handler.left
            self.view_left.set_image(iml)
        else:
            iml = self.image_handler.left
            self.view_left.set_image(iml)
            imr = self.image_handler.right
            self.view_right.set_image(imr)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import ctypes
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import cv2
import numpy as np
from tkinter import scrolledtext
from src.selection import launch_select_window
from src.mask import Mask
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import src.analysis
import json
from src.unet import UNet
import torch
import torch.nn.functional as F
from torchvision import transforms
from kornia.color import rgb_to_hsv

EMPTY_TENSOR = np.zeros((9, 9, 3),dtype=np.uint8)

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


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("TU Digital Electrochemistry Lab")
        self.geometry("1200x800")
        self.iconphoto(True, tk.PhotoImage(file='imgs/icon.png'))

        cm = open("data/spectrum.json")
        self.colormap = json.load(cm)

        self.image_handler = ImageWranger()
        self.interface = Interface(self)

        self.resizable = False


class SliderWindow:
    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Slider Window")
        def scale_callback(*args):
            self.master.analysis.discretize_heatmap(self.scale1.get())
            self.master.master.image_handler.image_select = 2
            self.master.master.image_handler.errors = self.master.analysis.error_image
            self.master.master.image_handler.right = self.master.analysis.error_image

            self.master.master.update_image_interface(2)
        # Create three Scale widgets (sliders)
        self.scale1 = tk.Scale(self.window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                               label="Threshold", command=scale_callback)

        # Create an "Update" button
        self.button = tk.Button(self.window, text="Update", command=self.update_values)

        # Pack the widgets
        self.scale1.pack()
        self.scale2.pack()
        self.scale3.pack()
        self.button.pack()

    def update_values(self):
        print("Value 1:", self.scale1.get())
        print("Value 2:", self.scale2.get())
        print("Value 3:", self.scale3.get())


class MaskAnalyzer:
    def __init__(self, master, image, material: str, auto_masked: bool):
        # Image is the colored mask, image_mask is the pure boolean
        self.master = master
        self.image = image
        self.material = material
        self.auto_masked = auto_masked
        self.error_mask = EMPTY_TENSOR
        # Error image not for scientific analysis. Visualization purposes only. Use error_mask
        self.error_image = EMPTY_TENSOR

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load("data/tudel_unet.1", map_location=torch.device(self.device)))


    # Alternative constructor for recalling object
    def set(self, image, material: str, auto_masked: bool):
        # Image is the colored mask, image_mask is the pure boolean
        self.image = image
        self.material = material
        self.auto_masked = auto_masked
        self.error_mask = EMPTY_TENSOR
        # Error image not for scientific analysis. Visualization purposes only. Use error_mask
        self.error_image = EMPTY_TENSOR
        self.neural_heatmap = EMPTY_TENSOR
        self.green = EMPTY_TENSOR

    
    def discretize_heatmap(self, threshold):
        omask = torch.where(torch.tensor(self.neural_heatmap) > threshold, torch.tensor(1.0), torch.tensor(0.0)).to(torch.uint8).numpy()
        self.error_mask = omask
        ori_sz = self.image.shape
        dst = cv2.bitwise_or(self.green, np.zeros(self.green.shape, dtype=np.uint8), mask=omask)
        resize_transform = transforms.Resize((ori_sz[0], ori_sz[1]))

        dst2 = torch.tensor(dst).permute((2,0,1))
        dst3 = resize_transform(dst2).permute((1,2,0))
        self.error_image = cv2.addWeighted(dst3.numpy(), 0.5, self.image, 0.7, 0)

    def gradient_segmentation(self):
        self.error_mask = src.analysis.errors(self.material, self.image, is_auto=self.auto_masked)
        print("errormask:  ", self.error_mask.shape)
        # Create the green segmentation dots
        green = np.zeros(self.image.shape, np.uint8)
        green[:, :, 1] = 255
        dst = cv2.bitwise_or(green, self.image, mask=self.error_mask)
        print("errorimg:  ", self.image.shape)
        print("green:  ", green.shape)
        self.error_image = cv2.addWeighted(dst, 0.5, self.image, 0.7, 0)
        dstgreen = cv2.bitwise_and(green, green, mask=self.error_mask)
        error_size = src.analysis.mask_size(dstgreen)
        deposit_size = src.analysis.mask_size(self.image)
        SingletonTextHandler.add_message(f"Percent Imperfection: {round((error_size/deposit_size)*100, 5)}%")

    def neural_segmentation(self, input_size=512, output_size=512):
        cv2.imwrite("test.png", self.image)
        ori_sz = self.image.shape
        t = transforms.Resize((input_size, input_size))(Image.fromarray(self.image))
        t = transforms.ToTensor()(t)

        # Convert the image to grayscale
        gray = cv2.cvtColor(t.permute((1,2,0)).numpy() * 255, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where black pixels are 1 and others are 0
        _, binary = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY_INV)
        #cv2.imshow("test", t.permute((1,2,0)).numpy())
        # Create a 7x7 kernel filled with ones
        kernel = np.ones((ori_sz[0] // 10, ori_sz[1] // 10), np.uint8)

        # Convolve the binary image with your kernel
        blackness_heatmap = cv2.filter2D(binary, -1, kernel)
        normalized_blackness_heatmap  = abs(1 - (blackness_heatmap / np.max(blackness_heatmap))) ** 2
        #
        # plt.imshow(blackness_heatmap)
        output = torch.zeros((input_size, input_size))
        self.model = self.model.to(self.device)
        g_img = rgb_to_hsv(t).permute((1,2,0))

        # Loop over the image patch by patch
        with torch.no_grad():
            for i in range(0, input_size, output_size):
                for j in range(0, input_size, output_size):
                    # Extract the patch
                    patch = g_img[i:i + output_size, j:j + output_size]

                    # Make sure it's the right shape
                    assert patch.shape == (output_size, output_size, 3)

                    # Add an extra batch dimension, and send patch to the same device as model
                    patch = patch.unsqueeze(0).to(self.device)

                    # Run the model on the patch
                    patch_output = self.model(patch.permute(0, 3, 1, 2))

                    # Remove the batch dimension and copy to CPU
                    patch_output = patch_output.squeeze(0).cpu()

                    # Make sure it's the right shape
                    patch_output = patch_output.squeeze(0)
                    assert patch_output.shape == (output_size, output_size)

                    # Insert the output patch into the output image
                    output[i:i + output_size, j:j + output_size] = patch_output


            green = np.zeros((output_size, output_size, 3), np.uint8)
            green[:, :, 1] = 255 
            self.green = green
            #cv2.imshow("normalized", normalized_blackness_heatmap)
            self.neural_heatmap = F.sigmoid(output.detach()).cpu() * normalized_blackness_heatmap
            self.discretize_heatmap(0.2)



class ImageWranger:
    def __init__(self):
        self.showmask = False
        self.hasmask = False
        # Init to arbitrary zero matricies (avoids possible future errors)
        self.left = EMPTY_TENSOR
        self.right = EMPTY_TENSOR
        self.original = EMPTY_TENSOR
        self.mask = EMPTY_TENSOR
        self.errors = EMPTY_TENSOR


        self.image_select = 0

    def index(self, index: int):
        # Because python doesn't have POINTERS or MATCH STATEMENTS LIKE ANY OTHER LANGUAGE
        if   index == 0: return self.original
        elif index == 1: return self.mask
        elif index == 2: return self.errors

    def select_image(self):
        imageselect = filedialog.askopenfilename(initialdir="Desktop",
                                             filetypes=[('Image files', '*.png'), ('Image files', '*.jpg'), ('Image files', '*.tif')])
        if not imageselect:
            return

        try:
            original = cv2.imread(imageselect)
            self.right = original
            self.left = original
            self.original = original
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
        self.mask = cropped
        self.right = cropped
        self.hasmask = True
        self.image_select = 1

    def auto_mask(self, material: str):
        lmask = Mask(material, self.right)
        self.mask = lmask.deposition_mask(self.right)
        dep_masked = cv2.bitwise_and(self.right, self.right, mask=self.mask)
        self.mask = dep_masked
        SingletonTextHandler.add_message(f"Auto masked image by {material} color range")
        self.hasmask = True
        self.right = self.mask
        self.image_select = 1

    def set_image(self, side: int, img_array):
        if side == 0:
            self.left = img_array
        elif side == 1:
            self.right = img_array
        else:
            raise Exception("Invalid side. Must be [0, 1]")


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


class Histogram(tk.Tk):
    def __init__(self, master):
        tk.Tk.__init__(self)
        self.title("Color Histogram")
        self.master = master


        # Create a Figure and a Canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack()

        self.button1 = tk.Button(self, text="Update", command=self.update_histogram)
        self.button2 = tk.Button(self, text="Save CSV", command=self.save_csv)
        self.button1.pack(side=tk.LEFT)
        self.button2.pack(side=tk.LEFT)
        # Start a thread that updates the histogram
        #threading.Thread(target=self.update_histogram, daemon=True).start()
        self.update_histogram()

    def save_csv(self):
        image = self.master.master.image_handler.right
        # Update the histogram
        self.ax.cla()  # Clear the plot
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        self.h, self.s, self.v = cv2.split(img_hsv)
        filh = self.h.flatten()[self.h.flatten() > 10].astype(np.uint8)
        fils = self.s.flatten()[self.s.flatten() > 10].astype(np.uint8)
        filv = self.v.flatten()[self.v.flatten() > 10].astype(np.uint8)
        data = np.stack((filh, fils, filv), axis=-1)
        filename = filedialog.asksaveasfilename(initialdir="Desktop", filetypes=[("TXT file", "*.csv")])
        np.savetxt(filename + ".csv", data, delimiter=",")

    def update_histogram(self):
        # Generate some random data
        image = self.master.master.image_handler.right
        # Update the histogram
        self.ax.cla()  # Clear the plot
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Split the HSV image into H, S, V arrays
        self.h, self.s, self.v = cv2.split(img_hsv)
        filh = self.h.flatten()[self.h.flatten() > 10]
        fils = self.s.flatten()[self.s.flatten() > 10]
        filv = self.v.flatten()[self.v.flatten() > 10]
        self.ax.hist(filh, bins=256, color='red', alpha=0.5, label='Hue', range=(0, 255))
        self.ax.hist(fils, bins=256, color='green', alpha=0.5, label='Saturation', range=(0, 255))
        self.ax.hist(filv, bins=256, color='blue', alpha=0.5, label='Value', range=(0, 255))
        self.ax.legend()
        # Redraw the canvas
        self.canvas.draw()

        # Wait a bit


class ExperimentInterface(ttk.Frame):
    def __init__(self, master, state='disabled'):
        super().__init__(master)
        self.master: Interface = master
        self.grid(padx=25, pady=25)
        self.material = "PbO2"
        self.state = state
        self.masked_state = 'normal' if self.master.image_handler.hasmask else 'disabled'
        self.analysis = MaskAnalyzer(self, self.master.image_handler.mask, self.material, self.master.image_handler.hasmask)

        self.create_widgets()



    def create_widgets(self):
        # ------------ Mask Widgets -----------
        self.mask_container = tk.Frame(self, width=200, height=500, relief=tk.RIDGE, borderwidth=3)
        self.mask_container.grid(row=0, column=0,  sticky=(tk.N, tk.S, tk.W))

        def change_material_callback(option: str):
            self.material = option

        choices = [choice for choice in self.master.master.colormap]
        option_variable = tk.StringVar(self)
        option_variable.set('Select Type')
        self.options = ttk.OptionMenu(self.mask_container, option_variable, *choices, command=lambda x: (change_material_callback(option_variable.get())))
        self.options.grid(row=0, column=0, padx=10, pady=10)

        self.man_mask = ttk.Button(self.mask_container, text="Manual Mask", state=self.state, command=lambda:
            (self.master.image_handler.manual_mask(),
             self.master.update_image_interface(2),
             self.master.console.update(),
             self.change_mask_state(True)))

        self.man_mask.grid(row=1, column=0, padx=10, pady=5)

        self.auto_mask = ttk.Button(self.mask_container, text="Auto Mask", state=self.state, command=lambda:
            (self.master.image_handler.auto_mask(self.material),
             self.master.update_image_interface(2),
             self.master.console.update(),
             self.change_mask_state(True)))

        self.auto_mask.grid(row=2, column=0, padx=10, pady=0)

        # ------------ Mask Widgets -----------
        self.a_container = tk.Frame(self, width=200, height=400, relief=tk.RIDGE, borderwidth=3)
        self.a_container.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W))

        def add_error_callback():
            self.master.image_handler.image_select = 2
            self.master.image_handler.errors = self.analysis.error_image
            self.master.image_handler.right = self.analysis.error_image

        self.leg_error = ttk.Button(self.a_container, text="Gradient\nSegmentation", state=self.masked_state, command=lambda:
        (
             self.analysis.set(self.master.image_handler.mask, self.material, self.master.image_handler.hasmask),
             self.analysis.gradient_segmentation(),
             add_error_callback(),
             self.master.console.update(),
             self.master.update_image_interface(2)),
        )
        self.leg_error.grid(row=1, column=0, padx=10, pady=5)

        self.leg_error = ttk.Button(self.a_container, text="Neural\nSegmentation", state=self.masked_state, command=lambda:
        (
             self.analysis.set(self.master.image_handler.mask, self.material, self.master.image_handler.hasmask),
             self.analysis.neural_segmentation(),
             SliderWindow(self),
             add_error_callback(),
             self.master.console.update(),
             self.master.update_image_interface(2)),
        )
        self.leg_error.grid(row=2, column=0, padx=10, pady=5)

        self.pixelarea = ttk.Button(self.a_container, text="Pixel Area", state=self.masked_state, command=lambda:
        (
            SingletonTextHandler.add_message(f"Masked area is {src.analysis.mask_size(self.master.image_handler.right)} pixels"),
            self.master.console.update(),
            self.master.update_image_interface(2)),
                                    )
        self.pixelarea.grid(row=3, column=0, padx=10, pady=0)


        # ------------ View Widgets -----------
        self.v_container = tk.Frame(self, width=200, height=400, relief=tk.RIDGE, borderwidth=3)
        self.v_container.grid(row=0, column=2, columnspan=1, sticky=(tk.N, tk.S, tk.E))

        self.leg_error = ttk.Button(self.v_container, text="Show\nHistogram", state=self.masked_state, command=lambda: (Histogram(self)))
        self.leg_error.grid(row=0, column=0, padx=10, pady=5)

        def hsvvals_callback():
            img_hsv = cv2.cvtColor(self.master.image_handler.right, cv2.COLOR_RGB2HSV)
            self.h, self.s, self.v = cv2.split(img_hsv)
            filh = self.h.flatten()[self.h.flatten() > 10].astype(np.uint8)
            fils = self.s.flatten()[self.s.flatten() > 10].astype(np.uint8)
            filv = self.v.flatten()[self.v.flatten() > 10].astype(np.uint8)
            SingletonTextHandler.add_message(f"Hue: {round(filh.mean(), 4)}")
            self.master.console.update()
            SingletonTextHandler.add_message(f"Sat: {round(fils.mean(), 4)}")
            self.master.console.update()
            SingletonTextHandler.add_message(f"Val: {round(filv.mean(), 4)}")
            self.master.console.update()
            SingletonTextHandler.add_message(f"---------")
            self.master.console.update()


        self.hsvvals = ttk.Button(self.v_container, text="HSV Vals", state=self.masked_state, command=hsvvals_callback)
        self.hsvvals.grid(row=1, column=0, padx=10, pady=0)

    def change_state(self, state: bool):
        self.state = 'normal' if state else 'disabled'

        self.create_widgets()

    def change_mask_state(self, state: bool):
        self.masked_state = 'normal' if state else 'disabled'

        self.create_widgets()


class Interface(ttk.Frame):
    def __init__(self, master: Application):
        super().__init__(master)
        self.master = master
        self.image_handler = master.image_handler
        self.grid(pady=25, padx=25)
        self.button_state = 'disabled'

        self.create_widgets()


    def change_state(self, state: bool):
        self.button_state = 'normal' if state else 'disabled'
        self.create_widgets()

    def create_widgets(self):
        # ------------- Image Boxes/Labels -------------
        # Labels for Left side
        self.label_left = ttk.Label(self, text="Original Image", font="bold")
        self.label_left.grid(row=0, column=0)
        # Bounding box of image
        self.label_view_left = tk.Label(self, text="Original", height="19", width="52", bd=0.5, relief="solid")
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

        # -------------- Show Mask ---------------
        def option_callback(num: int):
            self.image_handler.image_select = num
            self.update_image_interface(2)
        option_variable = tk.StringVar(self)
        option_variable.set('Original')
        choices = ["Original", "Mask", "Errors"]
        self.options = tk.OptionMenu(self, option_variable, *choices,
                                      command=lambda x: (option_callback(choices.index(option_variable.get()))))
        self.options.grid(row=2, column=1, padx=25, sticky=tk.E)

        # -------------- Buttons ---------------
        self.open_button = ttk.Button(self, text="Open Image", command=lambda:
            (self.image_handler.select_image(),
             self.change_state(True),
             self.update_image_interface(2),
             self.console.update()))
        self.open_button.grid(row=2, column=0, pady=10)

        self.save_button = ttk.Button(self, state=self.button_state, text="Save Image", command=lambda: (self.image_handler.write_image(), self.update_image_interface(2), self.console.update()))
        self.save_button.grid(row=2, column=1)

        # -------------- Experiments Interface ---------------
        self.divider = ttk.Separator(self, orient="vertical")
        self.divider.grid(row=2, column=0)

        self.experiments = ExperimentInterface(self)
        self.experiments.grid(column=2, row=1, sticky=(tk.N))

        # -------------- Console ---------------
        self.console = ConsoleView(self)
        self.console.grid(column=0, row=3, columnspan=2, sticky=(tk.N, tk.S, tk.W, tk.E))

    def update_image_interface(self, side: int):
        # Change button states depending on image loaded
        if self.image_handler.left is not None:
            self.save_button.config(state="normal")
            self.experiments.change_state(True)
        else:
            self.save_button.config(state="disabled")

        if side not in (0, 1, 2):
            raise Exception("Side must be (0, 1, 2)")

        if side == 1:
            # Get the regular image, unless the mask is selected
            imr = self.image_handler.index(self.image_handler.image_select)
            self.view_right.set_image(imr)
        elif side == 0:
            iml = self.image_handler.left
            self.view_left.set_image(iml)
        else:
            iml = self.image_handler.left
            self.view_left.set_image(iml)
            imr = self.image_handler.index(self.image_handler.image_select)
            self.view_right.set_image(imr)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
import tkinter as tk
from PIL import Image, ImageTk

WIDTH, HEIGHT = 900, 900
topx, topy, botx, boty = 0, 0, 0, 0
rect_id = None
canvas = None
output = (0, 0, 0, 0)



def get_mouse_posn(event):
    global topy, topx

    topx, topy = event.x, event.y

def update_sel_rect(event):
    global canvas
    global rect_id
    global output
    global topy, topx, botx, boty
    global x_top, y_top, x_bot, y_bot

    botx, boty = event.x, event.y
    canvas.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.

    output = (topx, topy, botx, boty)


def main(path="imgs/sample/perovskite.jpg"):
    global canvas, rect_id
    window = tk.Toplevel()
    window.title("Select Area")
    window.geometry('%sx%s' % (WIDTH, HEIGHT))
    window.configure(background='grey')

    img = ImageTk.PhotoImage(Image.open(path))
    canvas = tk.Canvas(window, width=img.width(), height=img.height(),
                       borderwidth=0, highlightthickness=0)
    canvas.pack(expand=True)
    canvas.img = img  # Keep reference in case this code is put into a function.
    canvas.create_image(0, 0, image=img, anchor=tk.NW)

    # Create selection rectangle (invisible since corner points are equal).
    rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                      dash=(2,2), fill='', outline='white')


    canvas.bind('<Button-1>', get_mouse_posn)
    canvas.bind('<B1-Motion>', update_sel_rect)  # Does what it sounds like

    # Use arbitrary variable var to force tkinter to wait until button click
    var = tk.IntVar()
    button = tk.Button(window, text="Save and Exit", width=20, height=5, command=lambda: var.set(1))
    button.place(relx=.1, rely=.1, anchor="c")

    button.wait_variable(var)
    window.destroy()
    return output
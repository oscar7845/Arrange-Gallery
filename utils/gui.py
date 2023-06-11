import tkinter as tk
from PIL import ImageTk, Image

class DraggableImage:
    def __init__(self, parent, image, row, column):
        self.parent = parent
        self.image = image
        self.row = row
        self.column = column

        self.label = tk.Label(parent, image=image)
        self.label.grid(row=row, column=column)

        self.label.bind("<Button-1>", self.start_drag)
        self.label.bind("<B1-Motion>", self.drag)
        self.label.bind("<ButtonRelease-1>", self.drop)

    def start_drag(self, event):
        self.dragging = True
        self.offset_x = event.x
        self.offset_y = event.y

    def drag(self, event):
        if self.dragging:
            x = self.label.winfo_x() - self.parent.winfo_rootx() + event.x - self.offset_x
            y = self.label.winfo_y() - self.parent.winfo_rooty() + event.y - self.offset_y
            self.label.place(x=x, y=y)

    def drop(self, event):
        self.dragging = False
        x, y = event.widget.winfo_rootx() - self.parent.winfo_rootx(), event.widget.winfo_rooty() - self.parent.winfo_rooty()
        target = self.parent.winfo_containing(x, y)
        if target != self.label and isinstance(target, tk.Label):
            target_row, target_column = target.grid_info()["row"], target.grid_info()["column"]
            print(f"Dropped image {self.row}, {self.column} onto {target_row}, {target_column}")

root = tk.Tk()

image_filenames = ["./images/free-pic1.jpg", "./images/free-pic2.jpg", "./images/free-pic3.jpg"]
images = [ImageTk.PhotoImage(Image.open(image_filename)) for image_filename in image_filenames]

draggable_images = []
for i, image in enumerate(images):
    row, column = i // 2, i % 2
    draggable_image = DraggableImage(root, image, row, column)
    draggable_images.append(draggable_image)

root.mainloop()
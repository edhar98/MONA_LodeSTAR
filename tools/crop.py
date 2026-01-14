import argparse
import os
import sys
import matplotlib

backends_to_try = ['Qt5Agg', 'Qt4Agg', 'TkAgg']
backend_set = False
backend_errors = []

for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        backend_set = True
        break
    except Exception as e:
        backend_errors.append(f"{backend}: {str(e)}")
        continue

if not backend_set:
    error_msg = "ERROR: No interactive backend available.\n"
    error_msg += f"Python: {sys.executable}\n"
    
    if backend_errors:
        error_msg += "Backend errors:\n" + "\n".join(f"  - {e}" for e in backend_errors)
    
    error_msg += "\nInstall PyQt5: pip install PyQt5"
    error_msg += "\n\nOr check DISPLAY variable: echo $DISPLAY"
    error_msg += "\nIf using SSH, use: ssh -X username@hostname"
    
    print(error_msg)
    sys.exit(1)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image
import numpy as np

class InteractiveCrop:
    def __init__(self, image_path: str, output_path: str):
        self.image_path = image_path
        self.output_path = output_path
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img)
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(self.img_array, cmap="gray" if len(self.img_array.shape) == 2 else None, origin='upper')
        self.ax.set_title("Draw a square box. Drag to move, scroll to resize, or click button to crop.")
        
        self.rect = None
        self.width_text = None
        self.diagonal1 = None
        self.diagonal2 = None
        self.center_circle = None
        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.min_size = 10
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        ax_crop = plt.axes([0.7, 0.02, 0.15, 0.04])
        self.btn_crop = Button(ax_crop, 'Crop & Save')
        self.btn_crop.on_clicked(self.crop_and_save)
        
        plt.subplots_adjust(bottom=0.1)
        plt.show()
    
    def get_rect_bounds(self):
        if self.rect is None:
            return None
        x = self.rect.get_x()
        y = self.rect.get_y()
        w = self.rect.get_width()
        h = self.rect.get_height()
        return (int(x), int(y), int(x + w), int(y + h))
    
    def is_point_in_rect(self, x: float, y: float) -> bool:
        if self.rect is None:
            return False
        rx = self.rect.get_x()
        ry = self.rect.get_y()
        rw = self.rect.get_width()
        rh = self.rect.get_height()
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def update_width_text(self):
        if self.rect is None:
            if self.width_text:
                self.width_text.remove()
                self.width_text = None
            return
        
        width = int(self.rect.get_width())
        center_x = self.rect.get_x() + self.rect.get_width() / 2
        top_y = self.rect.get_y()
        
        if self.width_text:
            self.width_text.remove()
        
        self.width_text = self.ax.text(
            center_x, top_y+5, f'{width}px',
            ha='center', va='bottom',
            color='yellow', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='yellow', linewidth=1),
            zorder=10
        )
    
    def update_decorations(self):
        if self.diagonal1:
            self.diagonal1.remove()
            self.diagonal1 = None
        if self.diagonal2:
            self.diagonal2.remove()
            self.diagonal2 = None
        if self.center_circle:
            self.center_circle.remove()
            self.center_circle = None
        
        if self.rect is None:
            return
        
        x = self.rect.get_x()
        y = self.rect.get_y()
        w = self.rect.get_width()
        h = self.rect.get_height()
        
        self.diagonal1, = self.ax.plot([x, x + w], [y, y + h], 'g-', linewidth=1, zorder=5)
        self.diagonal2, = self.ax.plot([x + w, x], [y, y + h], 'g-', linewidth=1, zorder=5)
        
        center_x = x + w / 2
        center_y = y + h / 2
        circle_radius = min(w, h) / 20
        self.center_circle = patches.Circle(
            (center_x, center_y), circle_radius,
            linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.8, zorder=6
        )
        self.ax.add_patch(self.center_circle)
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if self.rect is None:
            self.start_x = x
            self.start_y = y
            self.is_drawing = True
        elif self.is_point_in_rect(x, y):
            self.is_dragging = True
            rx = self.rect.get_x()
            ry = self.rect.get_y()
            self.drag_offset_x = x - rx
            self.drag_offset_y = y - ry
        else:
            self.start_x = x
            self.start_y = y
            self.is_drawing = True
            if self.rect:
                self.rect.remove()
                self.rect = None
            self.update_width_text()
            self.update_decorations()
    
    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        
        if self.is_drawing:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                self.is_drawing = False
                return
            
            size = max(abs(x - self.start_x), abs(y - self.start_y))
            if size < self.min_size:
                size = self.min_size
            
            center_x = (self.start_x + x) / 2
            center_y = (self.start_y + y) / 2
            
            if self.rect:
                self.rect.remove()
            
            self.rect = patches.Rectangle(
                (center_x - size/2, center_y - size/2),
                size, size,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            self.ax.add_patch(self.rect)
            self.update_width_text()
            self.update_decorations()
            self.fig.canvas.draw()
        
        self.is_drawing = False
        self.is_dragging = False
    
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if self.is_drawing and self.start_x is not None and self.start_y is not None:
            size = max(abs(x - self.start_x), abs(y - self.start_y))
            if size < self.min_size:
                size = self.min_size
            
            center_x = (self.start_x + x) / 2
            center_y = (self.start_y + y) / 2
            
            if self.rect:
                self.rect.remove()
            
            self.rect = patches.Rectangle(
                (center_x - size/2, center_y - size/2),
                size, size,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            self.ax.add_patch(self.rect)
            self.update_width_text()
            self.fig.canvas.draw()
        
        elif self.is_dragging and self.rect:
            new_x = x - self.drag_offset_x
            new_y = y - self.drag_offset_y
            
            img_height, img_width = self.img_array.shape[:2]
            size = self.rect.get_width()
            
            new_x = max(0, min(new_x, img_width - size))
            new_y = max(0, min(new_y, img_height - size))
            
            self.rect.set_x(new_x)
            self.rect.set_y(new_y)
            self.update_width_text()
            self.update_decorations()
            self.fig.canvas.draw()
    
    def on_scroll(self, event):
        if event.inaxes != self.ax or self.rect is None:
            return
        
        if not self.is_point_in_rect(event.xdata, event.ydata):
            return
        
        current_size = self.rect.get_width()
        scale_factor = 1.1 if event.button == 'up' else 0.9
        new_size = current_size * scale_factor
        
        if new_size < self.min_size:
            new_size = self.min_size
        
        img_height, img_width = self.img_array.shape[:2]
        center_x = self.rect.get_x() + current_size / 2
        center_y = self.rect.get_y() + current_size / 2
        
        half_size = new_size / 2
        new_x = max(0, min(center_x - half_size, img_width - new_size))
        new_y = max(0, min(center_y - half_size, img_height - new_size))
        
        self.rect.set_x(new_x)
        self.rect.set_y(new_y)
        self.rect.set_width(new_size)
        self.rect.set_height(new_size)
        self.update_width_text()
        self.update_decorations()
        self.fig.canvas.draw()
    
    def crop_and_save(self, event):
        bounds = self.get_rect_bounds()
        if bounds is None:
            print("No box drawn. Please draw a box first.")
            return
        
        x1, y1, x2, y2 = bounds
        
        x1 = max(0, min(x1, self.img.width))
        y1 = max(0, min(y1, self.img.height))
        x2 = max(0, min(x2, self.img.width))
        y2 = max(0, min(y2, self.img.height))
        
        if x2 <= x1 or y2 <= y1:
            print("Invalid crop area.")
            return
        
        cropped_img = self.img.crop((x1, y1, x2, y2))
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        cropped_img.save(self.output_path)
        print(f"Cropped image saved to: {self.output_path}")
        plt.close(self.fig)

def main():
    parser = argparse.ArgumentParser(description='Interactive image cropping tool')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    try:
        InteractiveCrop(args.input, args.output)
    except Exception as e:
        print(f"Error: Failed to open interactive window: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a display available (X11 forwarding if SSH)")
        print("2. Install tkinter: sudo apt-get install python3-tk")
        print("3. Set DISPLAY variable if needed: export DISPLAY=:0")
        print("4. Try alternative backend: MPLBACKEND=Qt5Agg python tools/crop.py ...")
        sys.exit(1)

if __name__ == '__main__':
    main()

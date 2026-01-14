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

class InteractiveMask:
    def __init__(self, image_path: str, output_path: str):
        self.image_path = image_path
        self.output_path = output_path
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img)
        
        self.phase = 'roi'
        self.roi_circle = None
        self.roi_center = None
        self.roi_radius = None
        self.noise_circle = None
        self.noise_center = None
        self.noise_radius = None
        self.noise_mean = None
        self.noise_std = None
        
        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.min_radius = 5
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(self.img_array, cmap="gray" if len(self.img_array.shape) == 2 else None, origin='upper')
        self.ax.set_title("Phase 1: Draw circle for ROI (region to keep). Drag to move, scroll to resize.")
        
        self.radius_text = None
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        ax_confirm = plt.axes([0.7, 0.02, 0.15, 0.04])
        self.btn_confirm = Button(ax_confirm, 'Confirm ROI')
        self.btn_confirm.on_clicked(self.on_confirm)
        
        plt.subplots_adjust(bottom=0.1)
        plt.show()
    
    def get_current_circle(self):
        if self.phase == 'roi':
            return self.roi_circle
        return self.noise_circle
    
    def set_current_circle(self, circle):
        if self.phase == 'roi':
            self.roi_circle = circle
        else:
            self.noise_circle = circle
    
    def get_current_center_radius(self):
        if self.phase == 'roi':
            return self.roi_center, self.roi_radius
        return self.noise_center, self.noise_radius
    
    def set_current_center_radius(self, center, radius):
        if self.phase == 'roi':
            self.roi_center = center
            self.roi_radius = radius
        else:
            self.noise_center = center
            self.noise_radius = radius
    
    def is_point_in_circle(self, x: float, y: float) -> bool:
        center, radius = self.get_current_center_radius()
        if center is None or radius is None:
            return False
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return dist <= radius
    
    def update_radius_text(self):
        if self.radius_text:
            self.radius_text.remove()
            self.radius_text = None
        
        center, radius = self.get_current_center_radius()
        if center is None or radius is None:
            return
        
        label = f'R={int(radius)}px'
        self.radius_text = self.ax.text(
            center[0], center[1] - radius - 10, label,
            ha='center', va='bottom',
            color='yellow', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='yellow', linewidth=1),
            zorder=10
        )
    
    def draw_circle(self, center, radius, color='r'):
        circle = patches.Circle(
            center, radius,
            linewidth=2, edgecolor=color, facecolor='none', zorder=5
        )
        self.ax.add_patch(circle)
        return circle
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        current_circle = self.get_current_circle()
        
        if current_circle is None:
            self.start_x = x
            self.start_y = y
            self.is_drawing = True
        elif self.is_point_in_circle(x, y):
            self.is_dragging = True
            center, _ = self.get_current_center_radius()
            self.drag_offset_x = x - center[0]
            self.drag_offset_y = y - center[1]
        else:
            self.start_x = x
            self.start_y = y
            self.is_drawing = True
            if current_circle:
                current_circle.remove()
                self.set_current_circle(None)
                self.set_current_center_radius(None, None)
            self.update_radius_text()
    
    def on_release(self, event):
        if self.is_drawing:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                self.is_drawing = False
                return
            
            radius = np.sqrt((x - self.start_x)**2 + (y - self.start_y)**2) / 2
            if radius < self.min_radius:
                radius = self.min_radius
            
            center = ((self.start_x + x) / 2, (self.start_y + y) / 2)
            
            current_circle = self.get_current_circle()
            if current_circle:
                current_circle.remove()
            
            color = 'r' if self.phase == 'roi' else 'cyan'
            new_circle = self.draw_circle(center, radius, color)
            self.set_current_circle(new_circle)
            self.set_current_center_radius(center, radius)
            self.update_radius_text()
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
            radius = np.sqrt((x - self.start_x)**2 + (y - self.start_y)**2) / 2
            if radius < self.min_radius:
                radius = self.min_radius
            
            center = ((self.start_x + x) / 2, (self.start_y + y) / 2)
            
            current_circle = self.get_current_circle()
            if current_circle:
                current_circle.remove()
            
            color = 'r' if self.phase == 'roi' else 'cyan'
            new_circle = self.draw_circle(center, radius, color)
            self.set_current_circle(new_circle)
            self.set_current_center_radius(center, radius)
            self.update_radius_text()
            self.fig.canvas.draw()
        
        elif self.is_dragging:
            center, radius = self.get_current_center_radius()
            if center is None:
                return
            
            new_center_x = x - self.drag_offset_x
            new_center_y = y - self.drag_offset_y
            
            img_height, img_width = self.img_array.shape[:2]
            new_center_x = max(radius, min(new_center_x, img_width - radius))
            new_center_y = max(radius, min(new_center_y, img_height - radius))
            
            new_center = (new_center_x, new_center_y)
            
            current_circle = self.get_current_circle()
            if current_circle:
                current_circle.remove()
            
            color = 'r' if self.phase == 'roi' else 'cyan'
            new_circle = self.draw_circle(new_center, radius, color)
            self.set_current_circle(new_circle)
            self.set_current_center_radius(new_center, radius)
            self.update_radius_text()
            self.fig.canvas.draw()
    
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        center, radius = self.get_current_center_radius()
        if center is None or radius is None:
            return
        
        if not self.is_point_in_circle(event.xdata, event.ydata):
            return
        
        scale_factor = 1.1 if event.button == 'up' else 0.9
        new_radius = radius * scale_factor
        
        if new_radius < self.min_radius:
            new_radius = self.min_radius
        
        img_height, img_width = self.img_array.shape[:2]
        new_radius = min(new_radius, center[0], center[1], img_width - center[0], img_height - center[1])
        
        current_circle = self.get_current_circle()
        if current_circle:
            current_circle.remove()
        
        color = 'r' if self.phase == 'roi' else 'cyan'
        new_circle = self.draw_circle(center, new_radius, color)
        self.set_current_circle(new_circle)
        self.set_current_center_radius(center, new_radius)
        self.update_radius_text()
        self.fig.canvas.draw()
    
    def on_confirm(self, event):
        if self.phase == 'roi':
            if self.roi_center is None or self.roi_radius is None:
                print("No ROI circle drawn. Please draw a circle first.")
                return
            
            self.phase = 'noise'
            self.roi_circle.set_edgecolor('green')
            self.ax.set_title("Phase 2: Draw circle for noise background region. Drag to move, scroll to resize.")
            self.btn_confirm.label.set_text('Apply Mask')
            self.fig.canvas.draw()
        
        else:
            if self.noise_center is None or self.noise_radius is None:
                print("No noise circle drawn. Please draw a circle first.")
                return
            
            self.apply_mask_and_save()
    
    def calculate_noise_stats(self):
        if self.noise_center is None or self.noise_radius is None:
            return
        
        h, w = self.img_array.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_noise_center = np.sqrt((x_coords - self.noise_center[0])**2 + (y_coords - self.noise_center[1])**2)
        noise_mask = dist_from_noise_center <= self.noise_radius
        
        if len(self.img_array.shape) == 3:
            noise_pixels = self.img_array[noise_mask].mean(axis=1)
        else:
            noise_pixels = self.img_array[noise_mask]
        
        self.noise_mean = np.mean(noise_pixels)
        self.noise_std = np.std(noise_pixels)
        print(f"Noise background - Mean: {self.noise_mean:.2f}, Std: {self.noise_std:.2f}")
    
    def apply_mask_and_save(self):
        self.calculate_noise_stats()
        
        h, w = self.img_array.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_roi_center = np.sqrt((x_coords - self.roi_center[0])**2 + (y_coords - self.roi_center[1])**2)
        roi_mask = dist_from_roi_center <= self.roi_radius
        
        if len(self.img_array.shape) == 3:
            masked_array = np.zeros_like(self.img_array)
            for c in range(self.img_array.shape[2]):
                masked_array[:, :, c] = np.where(roi_mask, self.img_array[:, :, c], 0)
        else:
            masked_array = np.where(roi_mask, self.img_array, 0)
        
        masked_img = Image.fromarray(masked_array.astype(self.img_array.dtype))
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        masked_img.save(self.output_path)
        print(f"Masked image saved to: {self.output_path}")
        print(f"ROI center: ({self.roi_center[0]:.1f}, {self.roi_center[1]:.1f}), radius: {self.roi_radius:.1f}")
        plt.close(self.fig)

def main():
    parser = argparse.ArgumentParser(description='Interactive image masking tool with circle selection')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    try:
        InteractiveMask(args.input, args.output)
    except Exception as e:
        print(f"Error: Failed to open interactive window: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a display available (X11 forwarding if SSH)")
        print("2. Install tkinter: sudo apt-get install python3-tk")
        print("3. Set DISPLAY variable if needed: export DISPLAY=:0")
        print("4. Try alternative backend: MPLBACKEND=Qt5Agg python tools/mask.py ...")
        sys.exit(1)

if __name__ == '__main__':
    main()


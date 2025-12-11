import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

PATH_TYPES = {"EPICYCLOID": 0, "HIPOCYCLOID": 1}
SHAPE_TYPES = {"WHEELS": 0, "GEARS": 1}

class PlotAnimation():
    def __init__(self, mode=[0, 0], base_rad=5, mov_rad=5, draw_rad=None, draw_offset=None):
        self.base_radius = base_rad
        self.mov_radius = mov_rad


        if draw_rad is None:
            self.draw_radius = mov_rad
            self.draw_offset = 0
        else:
            self.draw_radius = draw_rad * self.mov_radius
            self.draw_offset = draw_offset
        self.mode = {"PATH_TYPE": mode[0], "SHAPE_TYPE": mode[1]}
        
        self.dt = 0.02
        self.t = 0
        self.t_max = 100
        
        self.init_plot()

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        limit = 1.8 * (self.base_radius + self.mov_radius)
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        if self.mode["SHAPE_TYPE"] == SHAPE_TYPES["WHEELS"]:
            self.base_shape = Circle((0.0, 0.0), self.base_radius, fill=False, edgecolor='black', linewidth=2)
            self.mov_shape = Circle((0.0, 0.0), self.mov_radius, fill=False, edgecolor='blue', linewidth=2)
            self.ax.add_patch(self.base_shape)
            self.ax.add_patch(self.mov_shape)
            self.base_to_mov_ratio = self.base_radius / self.mov_radius
        else:
            self.teeth_height = 0.8  
            self.base_teeth_num = 20
            
            x, y = self.gear_curve(a=self.base_radius, b=20, n=self.base_teeth_num, h=self.teeth_height)
            verts = np.column_stack([x, y])
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            base_gear_path = Path(verts, codes)
            self.base_shape = PathPatch(base_gear_path, facecolor='#D3D3D3', edgecolor='black')

            ideal_N2 = self.base_teeth_num * (self.mov_radius / self.base_radius)
            self.mov_teeth_num = max(3, int(round(ideal_N2)))
            
            self.mov_radius = self.base_radius * (self.mov_teeth_num / self.base_teeth_num)
            self.base_to_mov_ratio = self.base_radius / self.mov_radius
            
            print(f"Adjusted Radii -> Base: {self.base_radius:.2f}, Moving: {self.mov_radius:.2f}")
            print(f"Teeth -> Base: {self.base_teeth_num}, Moving: {self.mov_teeth_num}")

            x, y = self.gear_curve(a=self.mov_radius, b=20, n=self.mov_teeth_num, h=self.teeth_height)
            verts = np.column_stack([x, y])
            codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
            move_gear_path = Path(verts, codes)
            self.mov_shape = PathPatch(move_gear_path, facecolor='#87CEEB', edgecolor='blue')
            
            self.ax.add_patch(self.base_shape)
            self.ax.add_patch(self.mov_shape)

        self.tracing_point, = self.ax.plot([], [], 'go', zorder=5, markersize=5) 
        self.mov_radius_plot, = self.ax.plot([], [], 'o-', color='black', alpha=0.5, markersize=3) 
        
        self.path_x_points = []
        self.path_y_points = []
        self.path, = self.ax.plot([], [], 'r-', linewidth=1.5) 
    
    def init_anim(self):
        return [self.mov_shape, self.tracing_point, self.mov_radius_plot, self.path]

    def update(self, frame):
        self.t += self.dt
        x, y, cx, cy = self.get_position(self.t)

        if self.t > self.t_max:
            self.t = 0
            self.path_x_points = []
            self.path_y_points = []
            self.path.set_data([], [])

        
        if self.mode["PATH_TYPE"] == PATH_TYPES["HIPOCYCLOID"]:
            angle_correction = np.pi 
            rotation_angle = self.t * (1 - self.base_to_mov_ratio) + angle_correction
        else:
            angle_correction = np.pi 
            rotation_angle = self.t * (1 + self.base_to_mov_ratio) + angle_correction

        transform = (
            Affine2D()
            .rotate_around(0, 0, rotation_angle) 
            .translate(cx, cy)
            + self.ax.transData
        )
        self.mov_shape.set_transform(transform) 

        self.tracing_point.set_data([x], [y])
        self.mov_radius_plot.set_data([x, cx], [y, cy])

        self.path_x_points.append(x)
        self.path_y_points.append(y)
        self.path.set_data(self.path_x_points, self.path_y_points)

        return [self.mov_shape, self.tracing_point, self.mov_radius_plot, self.path]

    def animate(self, frames=500, gen_gif=True):
        anim = FuncAnimation(self.fig, self.update, init_func=self.init_anim, 
                             frames=frames, interval=20, blit=True, repeat=True)
        if gen_gif == True:
            print("Saving GIF...")
            anim.save('spirograph.gif', writer='pillow', fps=30)
            print("GIF saved as 'spirograph.gif'")
        plt.show()
        return anim

    def get_position(self, t):
        R = self.base_radius
        r = self.mov_radius
        d = self.draw_radius
        off = self.draw_offset if self.draw_offset != None else 0
        
        if self.mode["PATH_TYPE"] == PATH_TYPES["HIPOCYCLOID"]:
            cx = (R - r) * np.cos(t)
            cy = (R - r) * np.sin(t)
            
            x = (R - r) * np.cos(t) + d * np.cos((R - r) / r * t + off)
            y = (R - r) * np.sin(t) - d * np.sin((R - r) / r * t + off)
        else:  
            cx = (R + r) * np.cos(t)
            cy = (R + r) * np.sin(t)
            
            x = (R + r) * np.cos(t) - d * np.cos((R + r) / r * t + off)
            y = (R + r) * np.sin(t) - d * np.sin((R + r) / r * t + off)
        
        return x, y, cx, cy
    
    def gear_curve(self, a, b, n, h, resolution=500):
        t = np.linspace(0, 2*np.pi, resolution)
        r = a + (h/2.0) * np.tanh(b * np.sin(n * t))
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        return x, y

if __name__ == "__main__":
    try:
        path_type = int(input("Choose path type - Epicycloid (0) / Hipocycloid (1): "))
        if path_type not in PATH_TYPES.values(): raise ValueError
        
        shape_type = int(input("Choose shape type - Wheels (0) / Gears (1): "))
        if shape_type not in SHAPE_TYPES.values(): raise ValueError
        
        base_shape_rad = float(input("Base radius (e.g., 10): "))
        moving_shape_rad = float(input("Moving radius (e.g., 3): "))

        mode = [path_type, shape_type]     
        animation = PlotAnimation(mode, base_rad=base_shape_rad, mov_rad=moving_shape_rad, draw_rad=0.5, draw_offset=20)
        animation.animate()
    except ValueError:
        print("Invalid input. Please enter numbers correctly.")

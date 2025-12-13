'''
Zrealizowane punkty:
1) convex hull
2) zmiana kierunku tygrysów klawiszem 'd'
'''


import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CONST_MAX_TIGER_SPEED = 0.5
CONST_MAP_SIZE = 20
CONST_MAP_MARGIN = 5

class Tiger():
    def __init__(self, id, speed = None):
        self.id = id
        self.x = random.choice(range(0,CONST_MAP_SIZE))
        self.y = random.choice(range(0,CONST_MAP_SIZE))
        self.speed = random.random() * CONST_MAX_TIGER_SPEED
        self.alpha = np.deg2rad(random.choice(range(0, 360)))
    
    def move(self):
        self.x = self.x + np.cos(self.alpha) * self.speed
        self.y = self.y + np.sin(self.alpha) * self.speed

        if self.x > CONST_MAP_SIZE or self.x < 0:
            self.alpha = np.pi - self.alpha
        if self.y > CONST_MAP_SIZE or self.y < 0:
            self.alpha = -self.alpha
    
    def get_position(self):
        return [self.x, self.y]
    
    def turn_around(self):
        self.alpha = self.alpha + np.pi



class Animation():
    def __init__(self, tigers: list):
        self.tigers = tigers
        self.init_x = [t.x for t in tigers]
        self.init_y = [t.y for t in tigers]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-CONST_MAP_MARGIN, CONST_MAP_SIZE + CONST_MAP_MARGIN)
        ax.set_ylim(-CONST_MAP_MARGIN, CONST_MAP_SIZE + CONST_MAP_MARGIN)
        self.scat = ax.scatter(self.init_x, self.init_y, c='orange', s=50, edgecolors='black', zorder=3)
        self.hull_line, = ax.plot([], [], 'b-', lw=2)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig = fig
        self.ax = ax
        pass
    
    def update(self, frame: int):
        tig_positions = []
        for tiger in self.tigers:
            tiger.move()
            tig_positions.append(tiger.get_position())
        
        tig_positions_np = np.array(tig_positions)
        convex_hull = self.graham_scan(tig_positions_np)
        if len(convex_hull) > 0:
            hull_to_draw = list(convex_hull)
            
            hull_to_draw.append(hull_to_draw[0])
            
            hull_np = np.array(hull_to_draw)
            
            self.hull_line.set_data(hull_np[:, 0], hull_np[:, 1])
        else:
            self.hull_line.set_data([], [])
        self.scat.set_offsets(tig_positions_np)
        return [self.scat, self.hull_line] 


    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()
    
    def graham_scan(self, points: np.ndarray):
            p_lowest = min(points.tolist(), key=lambda p: (p[1], p[0])) 
            p_lowest_np = np.array(p_lowest)
            
            idx_lowest = np.where((points == p_lowest_np).all(axis=1))[0][0]
            points_wo_ref = np.delete(points, idx_lowest, axis=0)
            
            sorted_points = self.polar_sort(p_lowest_np, points_wo_ref)

            hull_stack = [p_lowest, sorted_points[0], sorted_points[1]]

            for i in range(2, len(sorted_points)):
                next_point = sorted_points[i]

                while len(hull_stack) > 1 and self.graham_cross_product(hull_stack[-2], hull_stack[-1], next_point) <= 0:
                    hull_stack.pop()
                
                hull_stack.append(next_point)

            return hull_stack

    def polar_sort(self, ref_point: np.ndarray, points: np.ndarray):
        vectors = points - ref_point
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indecies = np.argsort(angles)
        sorted_points = points[sorted_indecies]
        
        return sorted_points
    
    def graham_cross_product(self, o, a, b):
            """
            oblicza iloczyn wektorowy dla wektorów OA i OB
            O - reference point (ostatni na stosie przed A)
            A - ostatni punkt na stosie
            B - nowy kandydat
            
            wynik:
            > 0 : skret w LEWO (wypukly = OK)
            < 0 : skret w PRAWO (wklesly = zle)
            = 0 : wspolionowe
            """
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def get_polar_angle(self, point: list, ref: list):
        dx = point[0] - ref[0]
        dy = point[1] - ref[1]
        return np.arctan2(dx, dy)
    
    def on_key_press(self, event):
        if event.key == 'd':
            print("Changing tiger direction")
            for tiger in self.tigers:
                tiger.turn_around()
        elif event.key == "escape":
            plt.close()




if __name__ == "__main__":
    tigers = [Tiger(i) for i in range(0,20)]
    anim = Animation(tigers)

    anim.animate()
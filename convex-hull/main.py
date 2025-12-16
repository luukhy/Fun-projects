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
CONST_MAP_MARGIN = 0

class Renderable():
    def __init__(self, id):
        self.id = id
        self.x = random.choice(range(0,CONST_MAP_SIZE))
        self.y = random.choice(range(0,CONST_MAP_SIZE))
    
    def draw():
        pass

    def get_position():
        pass

    

class Obstacle(Renderable):
    def __init__(self, id: int, shape: str = 'circle', radius: float = 3.0):
        super().__init__(id)
        if shape == 'circle':
            self.radius = radius 

    def get_position(self):
        return [self.x, self.y]
    
    def draw(self, ax):
        pass

class Tiger(Renderable):
    def __init__(self, id: int, speed = None, angle = None):
        super().__init__(id)
        self.speed = np.random.normal(CONST_MAX_TIGER_SPEED / 2, 0.1) if speed == None else speed
        self.speed = np.clip(self.speed, 0.1, CONST_MAX_TIGER_SPEED)
        self.alpha = np.deg2rad(random.choice(range(0, 360))) if angle == None else np.deg2rad(angle)
    
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
    
    def get_direction_vector(self):
        scalar = 2.5
        vec_u = scalar * self.speed * np.cos(self.alpha)
        vec_v = scalar * self.speed * np.sin(self.alpha)
        return vec_u, vec_v 

class Animation():
    def __init__(self, tigers: list, obstacles: list):
        self.tigers = tigers
        self.init_x = [t.x for t in tigers]
        self.init_y = [t.y for t in tigers]

        # U and V vectors represent the direction of a tiger
        self.init_u = [np.cos(t.alpha) for t in tigers]
        self.init_v = [np.sin(t.alpha) for t in tigers]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-CONST_MAP_MARGIN, CONST_MAP_SIZE + CONST_MAP_MARGIN)
        ax.set_ylim(-CONST_MAP_MARGIN, CONST_MAP_SIZE + CONST_MAP_MARGIN)

        self.scat = ax.scatter(self.init_x, self.init_y, c='orange', s=50, edgecolors='black', zorder=3)
        self.hull_line, = ax.plot([], [], 'b-', lw=2)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.quiver = ax.quiver(self.init_x, self.init_y, self.init_u, self.init_v,
                                color='black', width=0.005, scale=25, zorder=2)
        self.fig = fig
        self.ax = ax
        pass
    
    def update(self, frame: int):
        tig_positions = []
        vec_u = []
        vec_v = []
        for tiger in self.tigers:
            tiger.move()
            tig_positions.append(tiger.get_position())

            u, v = tiger.get_direction_vector()
            vec_u.append(u)
            vec_v.append(v)
        
        self.convex_hull(tig_positions)

        self.scat.set_offsets(tig_positions)
        self.quiver.set_offsets(tig_positions)
        self.quiver.set_UVC(vec_u, vec_v)
        return [self.scat, self.hull_line, self.quiver] 


    def animate(self):
        anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()
    
    def convex_hull(self, tig_positions: list):
        tig_positions_np = np.array(tig_positions)
        convex_hull = self.graham_scan(tig_positions_np)
        if len(convex_hull) > 0:
            hull_to_draw = list(convex_hull)
            
            hull_to_draw.append(hull_to_draw[0])
            
            hull_np = np.array(hull_to_draw)
            
            self.hull_line.set_data(hull_np[:, 0], hull_np[:, 1])
        else:
            self.hull_line.set_data([], [])

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

def tiger_assign_alpa():

    pass

def tiger_cdf(deg: int):
    
    pass

def generate_angles(how_many = 20, bias = 180.0, bias_scale = 1.0, distribution = None) -> list:
    distribution = 'normal' if distribution == None else distribution
    num_of_angs = 1000
    if distribution == 'normal':
        rand_deg_list = np.random.normal(bias, bias_scale, size = num_of_angs)
        rand_deg_list = np.abs(rand_deg_list % 360)
    elif distribution == 'uniform':
        rand_deg_list = np.random.randint(0, 360, size = num_of_angs)
    else:
        print("incorect type of distribution")
        return  []
    
    sorted_angs = np.sort(rand_deg_list)
    tigers_angles = []
    
    uninitialized_angs = how_many 
    rand_deg_idx = 0

    while(sorted_angs.size != 0 and uninitialized_angs != 0.0):
        value = rand_deg_list[rand_deg_idx]
        cdf_value = empirical_cdf_value(value, sorted_angs)
        num_to_init = int(uninitialized_angs * cdf_value)
        uninitialized_angs -= num_to_init

        tigers_angles.extend([value] * num_to_init)
        sorted_angs = np.delete(sorted_angs, np.where(sorted_angs == value))

        rand_deg_idx += 1
    return tigers_angles

def empirical_cdf_value(value: float, ndarr_sorted: np.ndarray):
    rank = np.searchsorted(ndarr_sorted, value, side = 'right')
    return rank / ndarr_sorted.size


if __name__ == "__main__":
    angles = generate_angles(bias = 180, bias_scale = 10)
    tigers = [Tiger(id=i, angle=angle) for i, angle in enumerate(angles)]
    anim = Animation(tigers, [])

    anim.animate()
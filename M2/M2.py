import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
import math
import tkinter as tk

def read_params(filename="params.txt"):
    params = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":")
                params[key.strip()] = val.strip()
    return params

def choose_dt(params):
    model = params.get("model", "s")
    deformation = int(params.get("def", 0))
    k = float(params.get("k", 1000.0))

    v0 = float(params.get("speed", 10))
    angle_deg = float(params.get("angle", 0))
    v1 = v0 * math.cos(math.radians(angle_deg))
    r1 = float(params.get("radius", 1.0))
    m1 = float(params.get("mass", 1.0))

    vmax = abs(v0)

    if model == "w":
        dt = r1 / (10 * max(vmax, 1e-6))


    elif model == "s":
        v02 = float(params.get("speed2", 0))
        angle2_deg = float(params.get("angle2", 180))
        v2 = v02 * math.cos(math.radians(angle2_deg))
        vmax = max(abs(v0), abs(v02), 1e-6)

        r2 = float(params.get("radius2", 1.0))
        m2 = float(params.get("mass2", 1.0))

        dt = min(r1, r2) / (10 * max(vmax,1e-6))
        

    return dt

def reflect_from_wall(v):
    vx, vy = v
    return np.array([-vx, vy])

def simulate_wall(params):
    v0 = float(params.get("speed", 10))
    angle_deg = float(params.get("angle", 45))
    duration = float(params.get("duration", 5.0))
    dt = choose_dt(params)
    steps = int(duration / dt)
    x0 = float(params.get("x0", 1.5))
    y0 = float(params.get("y0", 0.5))
    radius_sim = float(params.get("radius", 1.0))
    wall_x = 0.0
    m = float(params.get("mass", 1.0))

    k = float(params.get("k", 1000.0))

    deformation = int(params.get("def", 0))

    angle = math.radians(angle_deg)
    v = np.array([v0*math.cos(angle), v0*math.sin(angle)])
    x = np.array([x0, y0])

    traj, velocities, accelerations = [], [], []

    for _ in range(steps):
        delta = max(0.0, radius_sim - (x[0] - wall_x))
        a = np.array([0.0, 0.0])

        if deformation == 1:
            if delta > 0:
                a[0] = k * delta / m
        elif deformation == 2:
            if delta > 0:
                a[0] = k * delta**1.5 / m
        elif deformation == 0:
            if x[0] - radius_sim <= wall_x and v[0] < 0:
                v[0] = -v[0]
            delta = 0.0

        traj.append(x.copy())
        velocities.append(v.copy())
        accelerations.append(a.copy())

        v = v + a * dt
        x = x + v * dt

    return np.array(traj), np.array(velocities), np.array(accelerations), radius_sim

def animate_simulation(traj, velocities, accelerations, radius_sim, wall_x=0.0):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    fig, ax = plt.subplots(figsize=(screen_width/100, screen_height/100))

    margin = 0.1
    x_min, x_max = traj[:,0].min(), traj[:,0].max()
    y_min, y_max = traj[:,1].min(), traj[:,1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    min_range = 10

    if x_range < min_range:
        x_center = (x_min + x_max)/2
        x_min = x_center - min_range/2
        x_max = x_center + min_range/2
        x_range = min_range

    if y_range < min_range:
        y_center = (y_min + y_max)/2
        y_min = y_center - min_range/2
        y_max = y_center + min_range/2
        y_range = min_range

    ax.set_xlim(x_min - margin*x_range, x_max + margin*x_range)
    ax.set_ylim(y_min - margin*y_range, y_max + margin*y_range)

    ax.set_aspect('equal', adjustable='box')
    ax.axvline(wall_x, color="black", lw=2)

    ball = Circle((0,0), radius_sim, color="blue")
    ax.add_patch(ball)

    text_info = ax.text(0.02, 0.95, "", fontsize=12, transform=ax.transAxes)
    arrows = []

    def update(frame):
        x_pos = traj[frame]
        v = velocities[frame]
        a = accelerations[frame]

        ball.center = (x_pos[0], x_pos[1])

        for arr in arrows:
            arr.remove()
        arrows.clear()

        v_mod = np.linalg.norm(v)
        if v_mod > 1e-8:
            v_dir = v / v_mod
        else:
            v_dir = np.array([0.0, 0.0])

        arr_v = FancyArrowPatch(posA=(x_pos[0], x_pos[1]),
                                posB=(x_pos[0] + v_dir[0]*v_mod, x_pos[1] + v_dir[1]*v_mod),
                                arrowstyle='->', color='red', mutation_scale=15)
        
        a_mod = np.linalg.norm(a)
        if a_mod > 1e-8:
            a_dir = a / a_mod
        else:
            a_dir = np.array([0.0, 0.0])

        arr_a = FancyArrowPatch(posA=(x_pos[0], x_pos[1]),
                                posB=(x_pos[0] + a_dir[0]*a_mod, x_pos[1] + a_dir[1]*a_mod),
                                arrowstyle='->', color='green', mutation_scale=15)

        ax.add_patch(arr_v)
        ax.add_patch(arr_a)
        arrows.extend([arr_v, arr_a])

        v_mod = np.linalg.norm(v)
        a_mod = np.linalg.norm(a)
        v_ang = math.degrees(math.atan2(v[1], v[0])) if v_mod > 1e-8 else 0
        a_ang = math.degrees(math.atan2(a[1], a[0])) if a_mod > 1e-8 else 0

        text_info.set_text(
            f"v = {v_mod:.2f}, угол v = {v_ang:.1f}°\n"
            f"a = {a_mod:.2f}, угол a = {a_ang:.1f}°"
        )

        return [ball, text_info] + arrows

    ani = FuncAnimation(fig, update, frames=len(traj), interval=50, blit=False)
    plt.show()

def simulate_spheres(params):
    v0 = float(params.get("speed", 10))
    angle_deg = float(params.get("angle", 0))
    duration = float(params.get("duration", 5.0))
    dt = choose_dt(params)
    steps = int(duration / dt)

    x1 = np.array([float(params.get("x0", 0.0)), float(params.get("y0", 0.0))])
    r1 = float(params.get("radius", 1.0))
    m1 = float(params.get("mass", 1.0))
    v1 = np.array([v0*math.cos(math.radians(angle_deg)), v0*math.sin(math.radians(angle_deg))])

    x2 = np.array([float(params.get("x0_2", 5.0)), float(params.get("y0_2", 0.0))])
    r2 = float(params.get("radius2", 1.0))
    m2 = float(params.get("mass2", 1.0))
    v02 = float(params.get("speed2", 0.0))
    angle2_deg = float(params.get("angle2", 180))
    v2 = np.array([v02*math.cos(math.radians(angle2_deg)), v02*math.sin(math.radians(angle2_deg))])

    k = float(params.get("k", 1000.0))
    deformation = int(params.get("def", 0))

    traj1, traj2 = [], []
    vels1, vels2 = [], []
    accs1, accs2 = [], []

    for _ in range(steps):
        delta_v1 = np.array([0.0, 0.0])
        delta_v2 = np.array([0.0, 0.0])

        r_vec = x2 - x1
        dist = np.linalg.norm(r_vec)
        if dist > 1e-8:
            dir_vec = r_vec / dist
        else:
            dir_vec = np.array([1.0, 0.0])

        delta = max(0.0, (r1 + r2) - dist)

        if deformation in [1, 2] and delta > 0:
            if deformation == 1:
                F = k * delta
            elif deformation == 2:
                F = k * delta**1.5
            a1 = -F/m1 * dir_vec
            a2 = F/m2 * dir_vec
        elif deformation == 0 and delta > 0:
            v1n = np.dot(v1, dir_vec)
            v2n = np.dot(v2, dir_vec)
            v1n_new = (v1n*(m1-m2) + 2*m2*v2n)/(m1+m2)
            v2n_new = (v2n*(m2-m1) + 2*m1*v1n)/(m1+m2)
            v1 = v1 + (v1n_new - v1n)*dir_vec
            v2 = v2 + (v2n_new - v2n)*dir_vec
            a1 = np.array([0.0, 0.0])
            a2 = np.array([0.0, 0.0])
            delta = 0.0
        else:
            a1 = np.array([0.0, 0.0])
            a2 = np.array([0.0, 0.0])

        traj1.append(x1.copy())
        traj2.append(x2.copy())
        vels1.append(v1.copy())
        vels2.append(v2.copy())
        accs1.append(a1.copy())
        accs2.append(a2.copy())

        v1 = v1 + a1 * dt
        v2 = v2 + a2 * dt
        x1 = x1 + v1 * dt
        x2 = x2 + v2 * dt

    return (np.array(traj1), np.array(traj2),
            np.array(vels1), np.array(vels2),
            np.array(accs1), np.array(accs2),
            r1, r2)

def animate_spheres(traj1, traj2, vels1, vels2, accs1, accs2, r1, r2):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    fig, ax = plt.subplots(figsize=(screen_width/100, screen_height/100))

    margin = 0.1
    x_min = min(traj1[:,0].min(), traj2[:,0].min())
    x_max = max(traj1[:,0].max(), traj2[:,0].max())
    y_min = min(traj1[:,1].min(), traj2[:,1].min())
    y_max = max(traj1[:,1].max(), traj2[:,1].max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    min_range = 10

    if x_range < min_range:
        xc = (x_min + x_max)/2
        x_min, x_max = xc - min_range/2, xc + min_range/2
        x_range = min_range
    if y_range < min_range:
        yc = (y_min + y_max)/2
        y_min, y_max = yc - min_range/2, yc + min_range/2
        y_range = min_range

    ax.set_xlim(x_min - margin*x_range, x_max + margin*x_range)
    ax.set_ylim(y_min - margin*y_range, y_max + margin*y_range)
    ax.set_aspect('equal', adjustable='box')

    ball1 = Circle((0,0), r1, color="blue", alpha=0.6)
    ball2 = Circle((0,0), r2, color="orange", alpha=0.6)
    ax.add_patch(ball1)
    ax.add_patch(ball2)

    text_info = ax.text(0.02, 0.95, "", fontsize=12, transform=ax.transAxes)
    arrows = []

    def update(frame):
        x1, x2 = traj1[frame], traj2[frame]
        v1, v2 = vels1[frame], vels2[frame]
        a1, a2 = accs1[frame], accs2[frame]

        ball1.center = (x1[0], x1[1])
        ball2.center = (x2[0], x2[1])

        for arr in arrows:
            arr.remove()
        arrows.clear()

        def make_arrow(x, v, color):
            mod = np.linalg.norm(v)
            if mod > 1e-8:
                dir_vec = v / mod
            else:
                dir_vec = np.array([0.0, 0.0])
            return FancyArrowPatch(posA=(x[0], x[1]),
                                   posB=(x[0]+dir_vec[0]*mod, x[1]+dir_vec[1]*mod),
                                   arrowstyle='->', color=color, mutation_scale=15)

        arr_v1 = make_arrow(x1, v1, "red")
        arr_a1 = make_arrow(x1, a1, "green")
        arr_v2 = make_arrow(x2, v2, "red")
        arr_a2 = make_arrow(x2, a2, "green")

        for arr in [arr_v1, arr_a1, arr_v2, arr_a2]:
            ax.add_patch(arr)
            arrows.append(arr)

        text_info.set_text(
            f"Шар 1: v={np.linalg.norm(v1):.2f}, a={np.linalg.norm(a1):.2f}\n"
            f"Шар 2: v={np.linalg.norm(v2):.2f}, a={np.linalg.norm(a2):.2f}"
        )

        return [ball1, ball2, text_info] + arrows

    ani = FuncAnimation(fig, update, frames=len(traj1), interval=50, blit=False)
    plt.show()

#def validate_params(params):
  

def main():
    params = read_params("params.txt")
    model = params.get("model")

    if model == "w":
        traj, velocities, accelerations, radius_sim = simulate_wall(params)
        animate_simulation(traj, velocities, accelerations, radius_sim)
    elif model == "s":
        traj1, traj2, vels1, vels2, accs1, accs2, r1, r2 = simulate_spheres(params)
        animate_spheres(traj1, traj2, vels1, vels2, accs1, accs2, r1, r2)

if __name__ == "__main__":
    main()

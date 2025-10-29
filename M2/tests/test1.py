import numpy as np

from M2.M2 import simulate_wall, animate_simulation


def test_wall_reflection():
    params = {
        "model": "w",
        "speed": -10,
        "angle": 30,
        "duration": 2.0,
        "radius": 1.0,
        "mass": 1.0,
        "k": 1000,
        "x0": 3.5,
        "y0": 10.0
    }

    angle_rad = np.radians(params["angle"])
    v0 = params["speed"]
    analytical_vx = -v0 * np.cos(angle_rad)
    analytical_vy = v0 * np.sin(angle_rad)
    wall_x = 0.0

    for mode in [0, 1, 2]:
        params["def"] = mode
        if mode == 0:
            print("\n=== Аналитическое столкновение с стенкой ===", flush=True)
        elif mode == 1:
            print("\n=== Численное столкновение по закону Гука ===", flush=True)
        else:
            print("\n=== Численное столкновение по закону Герца ===", flush=True)

        traj, velocities, accelerations, radius_sim, dt = simulate_wall(params)
        v_final = velocities[-1]
        x_final = traj[-1]

        dev_vx = abs(v_final[0] - analytical_vx) if mode == 0 else 0.0
        dev_vy = abs(v_final[1] - analytical_vy) if mode == 0 else 0.0

        dev_vx_pct = (dev_vx / abs(analytical_vx) * 100) if analytical_vx != 0 else 0.0
        dev_vy_pct = (dev_vy / abs(analytical_vy) * 100) if analytical_vy != 0 else 0.0

        kinetic = 0.5 * params["mass"] * np.linalg.norm(v_final) ** 2
        delta = max(0.0, radius_sim - (x_final[0] - wall_x))
        if mode == 1:
            potential = 0.5 * params["k"] * delta ** 2
        elif mode == 2:
            potential = 2 / 5 * params["k"] * delta ** 2.5
        else:
            potential = 0.0
        total_energy = kinetic + potential

        print(f"Final position: x={x_final[0]:.3f}, y={x_final[1]:.3f}", flush=True)
        print(f"Final velocity (code solution): vx={v_final[0]:.3f}, vy={v_final[1]:.3f}", flush=True)
        print(f"Final velocity (analytical solution): vx={analytical_vx:.3f}, vy={analytical_vy:.3f}", flush=True)
        print(f"Deviation from analytical: dvx={dev_vx:.3f} ({dev_vx_pct:.2f}%), dvy={dev_vy:.3f} ({dev_vy_pct:.2f}%)")
        print(f"Total energy: {total_energy:.3f}", flush=True)
        animate_simulation(traj, velocities, accelerations, radius_sim, wall_x=0.0,
                           duration=float(params.get("duration", 5.0)), dt=dt)

    return traj, velocities, radius_sim


if __name__ == "__main__":
    test_wall_reflection()

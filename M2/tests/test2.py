import numpy as np

from M2.M2 import simulate_spheres, animate_spheres


def test_spheres_collision():
    params = {
        "model": "s",
        "speed": 10.0,
        "angle": 0.0,
        "speed2": 0.0,
        "angle2": 180.0,
        "duration": 2.0,
        "radius": 1.0,
        "mass": 1.0,
        "radius2": 1.0,
        "mass2": 1.0,
        "k": 1000,
        "x0": 10.0,
        "y0": 10.0,
        "x0_2": 20.0,
        "y0_2": 10.0
    }

    analytical_v1 = np.array([0.0, 0.0])
    analytical_v2 = np.array([10.0, 0.0])

    for mode in [0, 1, 2]:
        params["def"] = mode
        if mode == 0:
            print("\n=== Аналитическое столкновение шаров ===", flush=True)
        elif mode == 1:
            print("\n=== Численное столкновение по закону Гука ===", flush=True)
        else:
            print("\n=== Численное столкновение по закону Герца ===", flush=True)

        traj1, traj2, vels1, vels2, accs1, accs2, r1, r2, dt = simulate_spheres(params)
        v1_final = vels1[-1]
        v2_final = vels2[-1]
        x1_final = traj1[-1]
        x2_final = traj2[-1]

        dev_v1 = np.linalg.norm(v1_final - analytical_v1) if mode == 0 else 0.0
        dev_v2 = np.linalg.norm(v2_final - analytical_v2) if mode == 0 else 0.0

        dev_v1_pct = (dev_v1 / np.linalg.norm(analytical_v1) * 100) if np.linalg.norm(analytical_v1) != 0 else 0.0
        dev_v2_pct = (dev_v2 / np.linalg.norm(analytical_v2) * 100) if np.linalg.norm(analytical_v2) != 0 else 0.0

        kinetic = 0.5 * params["mass"] * np.linalg.norm(v1_final) ** 2 + 0.5 * params["mass2"] * np.linalg.norm(
            v2_final) ** 2
        delta = max(0.0, r1 + r2 - np.linalg.norm(x2_final - x1_final))
        if mode == 1:
            potential = 0.5 * params["k"] * delta ** 2
        elif mode == 2:
            potential = 2 / 5 * params["k"] * delta ** 2.5
        else:
            potential = 0.0
        total_energy = kinetic + potential

        print(
            f"Final positions: x1={x1_final[0]:.3f}, y1={x1_final[1]:.3f}; x2={x2_final[0]:.3f}, y2={x2_final[1]:.3f}")
        print(
            f"Final velocities (code solution): v1=({v1_final[0]:.3f}, {v1_final[1]:.3f}), v2=({v2_final[0]:.3f}, {v2_final[1]:.3f})")
        print(
            f"Final velocities (analytical solution): v1=({analytical_v1[0]:.3f}, {analytical_v1[1]:.3f}), v2=({analytical_v2[0]:.3f}, {analytical_v2[1]:.3f})")
        print(f"Deviation from analytical: dv1={dev_v1:.3f} ({dev_v1_pct:.2f}%), dv2={dev_v2:.3f} ({dev_v2_pct:.2f}%)")
        print(f"Total energy: {total_energy:.3f}", flush=True)

        animate_spheres(traj1, traj2, vels1, vels2, accs1, accs2, r1, r2, duration=float(params.get("duration", 5.0)),
                        dt=dt)

    return traj1, traj2, vels1, vels2, r1, r2


if __name__ == "__main__":
    test_spheres_collision()

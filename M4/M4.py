import json
import sys
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

def read_params():
    if not os.path.exists('M4/params.json'):
        raise FileNotFoundError("params.json not found")
    with open('M4/params.json', 'r') as f:
        return json.load(f)

def validate_params(params):
    mode = params.get('mode')
    if mode not in ['incline', 'horizontal']:
        raise ValueError("mode must be 'incline' or 'horizontal'")
    
    ball = params.get('ball', {})
    try:
        m = float(ball.get('mass'))
        r = float(ball.get('radius'))
        hollow = bool(ball.get('hollow'))
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid ball parameters") from e

    if not (0.001 <= m <= 1000.0):
        raise ValueError("mass must be in [0.001, 1000] kg")
    if not (0.001 <= r <= 2.0):
        raise ValueError("radius must be in [0.001, 2.0] m")

    I = (2/3)*m*r**2 if hollow else (2/5)*m*r**2

    if mode == 'incline':
        incline = params.get('incline', {})
        try:
            angle_deg = float(incline.get('angle_deg'))
            mu = float(incline.get('friction_coeff'))
        except (TypeError, ValueError) as e:
            raise ValueError("Invalid incline parameters") from e

        if not (0.0 <= angle_deg <= 90.0):
            raise ValueError("angle_deg must be in [0, 90] degrees")
        if not (0 <= mu <= 1.5):
            raise ValueError("friction_coeff must be in [0, 1.5] for dry friction")

        angle = np.radians(angle_deg)
        return {
            'mode': 'incline',
            'm': m, 'r': r, 'I': I,
            'mu': mu, 'angle': angle
        }
    else:
        plane = params.get('plane', {})
        try:
            mu = float(plane.get('friction_coeff'))
        except (TypeError, ValueError) as e:
            raise ValueError("Invalid plane parameters") from e

        if not (0 <= mu <= 1.5):
            raise ValueError("friction_coeff must be in [0, 1.5] for dry friction")

        init = params.get('initial_conditions', {})
        try:
            v0 = float(init.get('v0', 0.0))
            theta_v_deg = float(init.get('theta_v_deg', 0.0))
            omega0_x = float(init.get('omega0_x', 0.0))
            omega0_z = float(init.get('omega0_z', 0.0))
        except (TypeError, ValueError) as e:
            raise ValueError("Invalid initial conditions") from e

        if not (0.0 <= v0 <= 100.0):
            raise ValueError("v0 must be in [0, 100] m/s")
        if not (-1000.0 <= omega0_x <= 1000.0):
            raise ValueError("omega0_x must be in [-1000, 1000] rad/s")
        if not (-1000.0 <= omega0_z <= 1000.0):
            raise ValueError("omega0_z must be in [-1000, 1000] rad/s")

        v0x = v0 * np.cos(np.radians(theta_v_deg))
        v0z = v0 * np.sin(np.radians(theta_v_deg))

        return {
            'mode': 'horizontal',
            'm': m, 'r': r, 'I': I,
            'mu': mu,
            'v0x': v0x,
            'v0z': v0z,
            'omega0_x': omega0_x,
            'omega0_z': omega0_z
        }

def compute_derivatives(y, params):
    m, r, I, mu = params['m'], params['r'], params['I'], params['mu']
    g = 9.81

    if params['mode'] == 'incline':
        x_incline, v, omega = y
        angle = params['angle']
        N = m * g * np.cos(angle)
        f_max = mu * N
        v_rel = v - omega * r

        if abs(v_rel) < 1e-8:
            f_req = (m * g * np.sin(angle)) / (1 + m * r**2 / I)
            if abs(f_req) <= f_max:
                a = g * np.sin(angle) - f_req / m
                alpha = f_req * r / I
            else:
                a0 = g * np.sin(angle)
                if a0 > 0:
                    a = g * np.sin(angle) - f_max / m
                    alpha = f_max * r / I
                else:
                    a = g * np.sin(angle) + f_max / m
                    alpha = -f_max * r / I
        else:
            if v_rel > 0:
                a = g * np.sin(angle) - f_max / m
                alpha = f_max * r / I
            else:
                a = g * np.sin(angle) + f_max / m
                alpha = -f_max * r / I
        return v, a, alpha
    else:
        x, z, vx, vz, wx, wz = y
        N = m * g
        f_max = mu * N

        v_rel_x = vx + r * wz
        v_rel_z = vz - r * wx
        v_rel = np.array([v_rel_x, v_rel_z])
        v_rel_norm = np.linalg.norm(v_rel)

        if v_rel_norm < 1e-8:
            fx = fz = 0.0
            ax = az = 0.0
            alpha_x = alpha_z = 0.0
        else:
            f_dir = -v_rel / v_rel_norm
            fx, fz = f_max * f_dir
            ax = fx / m
            az = fz / m
            alpha_x = (-r * fz) / I
            alpha_z = ( r * fx) / I

        return vx, vz, ax, az, alpha_x, alpha_z

def simulate(params):
    g = 9.81
    dt = 1e-4
    t_max = 10.0
    t = 0.0

    if params['mode'] == 'incline':
        y = np.array([0.0, 0.0, 0.0])
    else:
        y = np.array([
            0.0,
            0.0,
            params['v0x'],
            params['v0z'],
            params['omega0_x'],
            params['omega0_z']
        ])

    ts = [t]
    ys = [y.copy()]

    steady_count = 0
    steady_threshold = 100

    while t < t_max:
        if params['mode'] == 'incline':
            v, a, alpha = compute_derivatives(y, params)
            y[0] += v * dt
            y[1] += a * dt
            y[2] += alpha * dt
        else:
            vx, vz, ax, az, alpha_x, alpha_z = compute_derivatives(y, params)
            y[0] += vx * dt
            y[1] += vz * dt
            y[2] += ax * dt
            y[3] += az * dt
            y[4] += alpha_x * dt
            y[5] += alpha_z * dt

        t += dt
        ts.append(t)
        ys.append(y.copy())

        if params['mode'] == 'horizontal':
            vx, vz = y[2], y[3]
            wx, wz = y[4], y[5]
            v_rel_x = vx + params['r'] * wz
            v_rel_z = vz - params['r'] * wx
            v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_z**2)
            v_cm_norm = np.sqrt(vx**2 + vz**2)
            if v_rel_norm < 1e-6 and v_cm_norm < 1e-6:
                steady_count += 1
            else:
                steady_count = 0
        else:
            v_rel = abs(y[1] - y[2] * params['r'])
            if v_rel < 1e-6:
                steady_count += 1
            else:
                steady_count = 0

        if steady_count > steady_threshold:
            break

    ts = np.array(ts)
    ys = np.array(ys)

    if params['mode'] == 'incline':
        x_incline = ys[:, 0]
        x_xy = x_incline * np.cos(params['angle'])
        y_xy = -x_incline * np.sin(params['angle'])
        E_pot = params['m'] * g * (-y_xy)
        v = ys[:, 1]
        omega = ys[:, 2]
        E_trans = 0.5 * params['m'] * v**2
        E_rot = 0.5 * params['I'] * omega**2
    else:
        x_xy = ys[:, 0]
        y_xy = ys[:, 1]
        vx, vz = ys[:, 2], ys[:, 3]
        wx, wz = ys[:, 4], ys[:, 5]
        v = np.sqrt(vx**2 + vz**2)
        omega = np.sqrt(wx**2 + wz**2)
        E_trans = 0.5 * params['m'] * (vx**2 + vz**2)
        E_rot = 0.5 * params['I'] * (wx**2 + wz**2)
        E_pot = np.zeros_like(ts)

    return ts, x_xy, y_xy, v, omega, E_trans, E_rot, E_pot

def plot_results(t, x, z, v, omega, E_trans, E_rot, E_pot):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0,0].plot(t, x, label='x')
    axs[0,0].plot(t, z, label='z')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Position (m)')
    axs[0,0].set_title('Position vs Time (x and z)')
    axs[0,0].legend()
    
    axs[0,1].plot(t, v, label='|v| (m/s)')
    axs[0,1].plot(t, omega, label='|ω| (rad/s)')
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Speed / Angular speed')
    axs[0,1].set_title('Speeds')
    axs[0,1].legend()
    
    axs[1,0].plot(t, E_trans, label='Translational KE')
    axs[1,0].plot(t, E_rot, label='Rotational KE')
    axs[1,0].plot(t, E_pot, label='Potential Energy')
    axs[1,0].set_xlabel('Time (s)')
    axs[1,0].set_ylabel('Energy (J)')
    axs[1,0].set_title('Energies')
    axs[1,0].legend()
    
    E_total = E_trans + E_rot + E_pot
    axs[1,1].plot(t, E_total, label='Total Energy', color='black')
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Energy (J)')
    axs[1,1].set_title('Total Energy')
    axs[1,1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        raw = read_params()
        params = validate_params(raw)
        print("Starting simulation...")
        t, x, z, v, omega, E_trans, E_rot, E_pot = simulate(params)
        print(f"Simulation finished. Total time: {t[-1]:.3f} s, steps: {len(t)}")
        print(f"Final |v| = {v[-1]:.2f} m/s, |ω| = {omega[-1]:.2f} rad/s")
        plot_results(t, x, z, v, omega, E_trans, E_rot, E_pot)
    except Exception as e:
        print(f"Error: {e}")

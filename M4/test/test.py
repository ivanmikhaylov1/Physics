import unittest

import numpy as np

from M4.M4 import simulate, validate_params


class TestInclineAnalytical(unittest.TestCase):
    def test_no_slip(self):
        m = np.random.uniform(0.001, 1000.0)
        r = np.random.uniform(0.001, 2.0)
        hollow = np.random.choice([True, False])
        angle_deg = np.random.uniform(5.0, 90.0)
        I = (2.0 / 3.0) * m * r * r if hollow else (2.0 / 5.0) * m * r * r
        angle = np.radians(angle_deg)
        mu_min = np.tan(angle) / (1.0 + I / (m * r * r))
        fc = float(np.random.uniform(max(0.25, mu_min + 1e-4), 0.6))

        g = 9.81
        a_analytical = g * np.sin(angle) / (1.0 + I / (m * r * r))

        params = {
            "mode": "incline",
            "ball": {
                "mass": m,
                "radius": r,
                "hollow": hollow
            },
            "incline": {
                "angle_deg": angle_deg,
                "friction_coeff": fc
            }
        }

        validated = validate_params(params)
        ts, x_num, y_num, v_num, omega_num, E_trans, E_rot, E_pot = simulate(validated)

        v_rel = np.abs(v_num - omega_num * r)
        self.assertTrue(np.all(v_rel < 1e-2), f"Не выполнено качение без проскальзывания, max v_rel={v_rel.max():.6f}")

        v_analytical = a_analytical * ts
        x_analytical = 0.5 * a_analytical * ts ** 2
        omega_analytical = v_analytical / r

        np.testing.assert_allclose(v_num, v_analytical, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(omega_num, omega_analytical, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(x_num, x_analytical, rtol=5e-3, atol=1e-3)

    def test_with_slip(self):
        diagnostics = []
        for attempt in range(12):
            m = np.random.uniform(0.001, 1000.0)
            r = np.random.uniform(0.001, 2.0)
            hollow = np.random.choice([True, False])
            fc = np.random.uniform(0.001, 1.5)
            v0 = np.random.uniform(0.001, 100.0)
            theta_deg = np.random.uniform(0.0, 90.0)
            omega_target = -v0 / r
            omega0_x = np.random.uniform(omega_target * 0.98, omega_target * 1.02)
            omega0_z = np.random.uniform(omega_target * 0.98, omega_target * 1.02)

            params = {
                "mode": "horizontal",
                "ball": {
                    "mass": m,
                    "radius": r,
                    "hollow": hollow
                },
                "plane": {
                    "friction_coeff": fc
                },
                "initial_conditions": {
                    "v0": v0,
                    "theta_v_deg": float(theta_deg),
                    "omega0_x": float(omega0_x),
                    "omega0_z": float(omega0_z)
                }
            }

            try:
                validated = validate_params(params)
            except Exception as e:
                diagnostics.append(("validate_error", params, str(e)))
                continue

            ts, x_num, y_num, v_num, omega_num, E_trans, E_rot, E_pot = simulate(validated)

            total_energy = E_trans + E_rot + E_pot
            energy_ok = bool(total_energy[-1] <= total_energy[0] + 1e-3)

            rel_speed = np.abs(v_num + omega_num * r)
            n = len(rel_speed)
            k = max(1, n // 10)
            rel_start = float(np.mean(rel_speed[:k]))
            rel_end = float(np.mean(rel_speed[-k:]))

            threshold = max(0.04, rel_start * 0.85)
            passed = energy_ok and (rel_end <= threshold)

            if passed:
                return

        last_msgs = []
        for tag, p, info in diagnostics[-6:]:
            last_msgs.append(f"{tag} params={p} info={info}")


if __name__ == "__main__":
    unittest.main()

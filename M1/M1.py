import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import yaml


with open("M1/params.yml", "r") as f:
  params = yaml.safe_load(f)

### Начальные условия

g = 9.8
v0 = params["Начальная скорость"]
angle_degree = params["Угол броска"]
angle_radians = np.radians(angle_degree)
k_linear = params["Линейное сопротивление"]
k_quadratic = params["Квадратичное сопротивление"]
x0 = params["Начальная позиция x"]
y0 = params["Начальная позиция y"]

if v0 > 2000:
  raise ValueError("Слишком большая скорость")
if v0 <= 0:
  raise ValueError("Скорость должна быть положительной")
if angle_degree > 90:
  raise ValueError("Угол должен быть не больше 90 градусов")
if angle_degree <= 0:
  raise ValueError("Значение угла должно быть положительным")
if k_linear >= 0:
  raise ValueError("Значение линейного сопротивления должно быть положительным")
if k_quadratic >= 0:
  raise ValueError("Значение квадратичного сопротивления должно быть положительным")

### Проекции скоростей

def model(Y, t, g, k, resistance_type):
  x, y, vx, vy = Y

  if resistance_type == "Нет":
    ax = 0.0
    ay = -g
  elif resistance_type == "Линейное":
    ax = - k * vx
    ay = - g - k * vy
  elif resistance_type == "Квадратичное":
    v = np.sqrt(vx ** 2 + vy ** 2)
    ax = - k * v * vx
    ay = - g - k * v * vy

  return [vx, vy, ax, ay]

def calculate_trace(x0, y0, v0, angle_degree, g, k, resistance_type):
    angle_radians = np.radians(angle_degree)
    Y0 = [x0, y0, v0 * np.cos(angle_radians), v0 * np.sin(angle_radians)]

    t_flight_estimate = 2 * Y0[3] / g
    t_points = np.linspace(0, t_flight_estimate * 2.0, 1000)

    args_tuple = (g, k, resistance_type)
    solution = odeint(model, Y0, t_points, args=args_tuple)

    for i in range(len(solution)):
        if solution[i, 1] < 0:
            return solution[:i]

    return solution

### Расчеты и визуализация

trajectory_none = calculate_trace(x0, y0, v0, angle_degree, g, 0, "Нет")
trajectory_linear = calculate_trace(x0, y0, v0, angle_degree, g, k_linear, "Линейное")
trajectory_quadratic = calculate_trace(x0, y0, v0, angle_degree, g, k_quadratic, "Квадратичное")

t_theory_max = 2 * v0 * np.sin(np.radians(angle_degree)) / g
t_theory = np.linspace(0, t_theory_max, 200)
x_theory = v0 * np.cos(np.radians(angle_degree)) * t_theory
y_theory = v0 * np.sin(np.radians(angle_degree)) * t_theory - 0.5 * g * t_theory**2

print(f"Параметры: v0={v0} м/с, угол={angle_degree}°")

# Сравнение с теорией
range_numeric = trajectory_none[-1, 0]
range_theory = (v0**2 * np.sin(2 * np.radians(angle_degree))) / g
print(f"Дальность (без сопр., численно): {range_numeric:.2f} м")
print(f"Дальность (без сопр., теория):   {range_theory:.2f} м")
print(f"-> Ошибка численного метода: {abs(range_numeric - range_theory)/range_theory*100:.4f} %")
print("-" * 40)

# Результаты для моделей с сопротивлением
print(f"Дальность (линейное сопр., k={k_linear}):    {trajectory_linear[-1, 0]:.2f} м")
print(f"Дальность (квадратичное сопр., k={k_quadratic}): {trajectory_quadratic[-1, 0]:.2f} м")

# Построение графика
plt.figure(figsize=(14, 8))
plt.plot(trajectory_none[:, 0], trajectory_none[:, 1], 'b-', linewidth=2, label='Численно (без сопр.)')
plt.plot(x_theory, y_theory, 'r--', linewidth=2, label='Теория (без сопр.)')
plt.plot(trajectory_linear[:, 0], trajectory_linear[:, 1], 'g-', label=f'Линейное сопр. (k={k_linear})')
plt.plot(trajectory_quadratic[:, 0], trajectory_quadratic[:, 1], 'm-', label=f'Квадратичное сопр. (k={k_quadratic})')

plt.title('Сравнение моделей полёта камня', fontsize=16)
plt.xlabel('Дальность, м', fontsize=12)
plt.ylabel('Высота, м', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.axis('equal')
plt.savefig('trajectory.png', bbox_inches='tight')
plt.close()

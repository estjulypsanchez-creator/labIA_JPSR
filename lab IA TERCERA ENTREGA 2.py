"""
pendulo_asdw.py
Péndulo invertido sobre carro con controlador difuso Mamdani
y control manual por teclas A D S W.

Cómo usar:
 - Ejecuta: python pendulo_asdw.py
 - Se abrirá una ventana gráfica; usa las teclas:
     A = fuerza hacia la izquierda (incrementa negativo)
     D = fuerza hacia la derecha (incrementa positivo)
     S = poner fuerza manual a 0
     W = alternar modo: AUTO -> ASSIST -> MANUAL
 - Cierra la ventana para detener la simulación.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
import time

# ---------------------- Dinámica: CartPole ----------------------
class CartPole:
    def __init__(self, M=1.0, m=0.15, l=0.5, g=9.81, dt=0.02, force_max=15.0):
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt
        self.force_max = force_max

    def derivatives(self, state, F):
        x, x_dot, theta, theta_dot = state
        m = self.m; M = self.M; l = self.l; g = self.g
        sin_t = np.sin(theta); cos_t = np.cos(theta)
        total = m + M
        polemass_length = m * l
        temp = (F + polemass_length * theta_dot**2 * sin_t) / total
        theta_acc = (g * sin_t - cos_t * temp) / (l * (4.0/3.0 - m * cos_t**2 / total))
        x_acc = temp - polemass_length * theta_acc * cos_t / total
        return np.array([x_dot, x_acc, theta_dot, theta_acc])

    def step(self, state, F):
        F = np.clip(F, -self.force_max, self.force_max)
        dt = self.dt
        k1 = self.derivatives(state, F)
        k2 = self.derivatives(state + 0.5 * dt * k1, F)
        k3 = self.derivatives(state + 0.5 * dt * k2, F)
        k4 = self.derivatives(state + dt * k3, F)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        new_state[2] = (new_state[2] + np.pi) % (2*np.pi) - np.pi
        return new_state

# ---------------------- Fuzzy utils ----------------------
def triangular(x, a, b, c):
    if a == b and b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def fuzzify(value, terms):
    return {name: triangular(value, *params) for name, params in terms.items()}

# ---------------------- Mamdani Fuzzy Controller ----------------------
class MamdaniFuzzyController:
    def __init__(self, angle_range_deg=30.0, angvel_range_deg=200.0, force_max=15.0, scale_out=1.0):
        self.angle_range = angle_range_deg
        self.angvel_range = angvel_range_deg
        self.force_max = force_max
        self.scale_out = scale_out
        a = angle_range_deg; v = angvel_range_deg; f = force_max * scale_out

        self.angle_terms = {
            'NL': (-a*3, -a, -a/2),
            'NS': (-a, -a/2, 0.0),
            'Z' : (-a/10, 0.0, a/10),
            'PS': (0.0, a/2, a),
            'PL': (a/2, a, a*3)
        }

        self.angvel_terms = {
            'NL': (-v*3, -v, -v/2),
            'NS': (-v, -v/2, 0.0),
            'Z' : (-v/10, 0.0, v/10),
            'PS': (0.0, v/2, v),
            'PL': (v/2, v, v*3)
        }

        self.force_terms = {
            'NL': (-f*2, -f, -f/2),
            'NS': (-f, -f/2, 0.0),
            'Z' : (-f*0.1, 0.0, f*0.1),
            'PS': (0.0, f/2, f),
            'PL': (f/2, f, f*2)
        }

        mapping = {'NL':'NL','NS':'NS','Z':'Z','PS':'PS','PL':'PL'}
        self.rules = [(at, vt, mapping[at]) for at in self.angle_terms.keys() for vt in self.angvel_terms.keys()]

    def compute(self, theta_rad, theta_dot_rad):
        theta_deg = np.degrees(theta_rad)
        theta_dot_deg = np.degrees(theta_dot_rad)
        a_mf = fuzzify(theta_deg, self.angle_terms)
        v_mf = fuzzify(theta_dot_deg, self.angvel_terms)

        output_universe = np.linspace(-self.force_max*self.scale_out, self.force_max*self.scale_out, 201)
        agg = np.zeros_like(output_universe)

        for (at, vt, out) in self.rules:
            alpha = min(a_mf[at], v_mf[vt])
            if alpha <= 0.0: continue
            out_params = self.force_terms[out]
            out_memb = np.array([triangular(x, *out_params) for x in output_universe])
            agg = np.maximum(agg, np.minimum(alpha, out_memb))

        if agg.sum() == 0:
            crisp = 0.0
        else:
            crisp = (agg * output_universe).sum() / agg.sum()
        return float(np.clip(crisp, -self.force_max*self.scale_out, self.force_max*self.scale_out))

# ---------------------- Control manual / modos ----------------------
# Variables globales de control manual
current_manual_force = 0.0
# modos: 'AUTO' (solo difuso), 'ASSIST' (difuso + manual), 'MANUAL' (solo manual)
control_mode = 'AUTO'

def toggle_mode():
    global control_mode
    if control_mode == 'AUTO':
        control_mode = 'ASSIST'
    elif control_mode == 'ASSIST':
        control_mode = 'MANUAL'
    else:
        control_mode = 'AUTO'
    print(f"Modo cambiado: {control_mode}")

# ---------------------- Simulación interactiva (visual en tiempo real) ----------------------
def run_interactive(sim_time=60.0):
    global current_manual_force, control_mode

    # entorno y controlador
    env = CartPole()
    controller = MamdaniFuzzyController(angle_range_deg=25.0, angvel_range_deg=200.0, force_max=env.force_max, scale_out=1.0)

    # estado inicial (x, x_dot, theta, theta_dot)
    state = np.array([0.0, 0.0, np.radians(6.0), 0.0])
    dt = env.dt
    steps = int(sim_time / dt)

    # figura
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    cart_width = 0.4
    pole_len = env.l * 2.0

    cart_patch = plt.Rectangle((state[0]-cart_width/2, -0.1), cart_width, 0.2, ec='k', fc='0.8')
    pole_line, = ax.plot([], [], lw=3)
    bob, = ax.plot([], [], 'o', ms=8)
    txt_mode = ax.text(-2.4, 1.2, f"Modo: {control_mode}", fontsize=10)
    txt_force = ax.text(-2.4, 1.0, f"Fuerza manual: {current_manual_force:.2f}", fontsize=10)
    txt_angle = ax.text(-2.4, 0.8, f"Ang (deg): {np.degrees(state[2]):+.2f}", fontsize=10)

    ax.add_patch(cart_patch)

    # Key handler
    step_force = 3.0  # cuánto cambia la fuerza cada pulsación A/D
    def on_key(event):
        global current_manual_force, control_mode
        key = event.key
        if key is None:
            return
        key = key.lower()
        if key == 'a':
            current_manual_force -= step_force
        elif key == 'd':
            current_manual_force += step_force
        elif key == 's':
            current_manual_force = 0.0
        elif key == 'w':
            toggle_mode()
        # clip manual force
        current_manual_force = float(np.clip(current_manual_force, -env.force_max, env.force_max))
        print(f"[tecla {key}] Fuerza manual={current_manual_force:.2f}  Modo={control_mode}")

    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        for tstep in range(steps):
            # calcular fuerza de difuso
            fuzzy_F = controller.compute(state[2], state[3])

            # seleccionar fuerza final según modo
            if control_mode == 'AUTO':
                F = fuzzy_F
            elif control_mode == 'ASSIST':
                # suma de ambos (asistencia)
                F = fuzzy_F + current_manual_force
            else:  # MANUAL
                F = current_manual_force

            # aplicar dinámica
            state = env.step(state, F)

            # actualizar figura
            cart_x = state[0]
            cart_patch.set_x(cart_x - cart_width/2)
            theta = state[2]
            x0 = cart_x
            y0 = 0.0 + 0.1
            x1 = x0 + pole_len * np.sin(theta)
            y1 = y0 + pole_len * np.cos(theta)
            pole_line.set_data([x0, x1], [y0, y1])
            bob.set_data([x1], [y1])

            txt_mode.set_text(f"Modo: {control_mode}")
            txt_force.set_text(f"Fuerza manual: {current_manual_force:.2f}")
            txt_angle.set_text(f"Ang (deg): {np.degrees(state[2]):+.2f}")

            fig.canvas.draw()
            plt.pause(dt)  # pequeña espera para actualizar ventana

            # condición de seguridad opcional: si pierdes completamente (ángulo muy grande), reinicia ángulo pequeño
            if abs(state[2]) > np.radians(170):
                print("Ángulo extremo detectado: reiniciando ángulo a 6°")
                state[2] = np.radians(6.0)
                state[3] = 0.0

    except KeyboardInterrupt:
        print("Interrumpido por usuario.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    print("Controles: A (izq), D (der), S (reset fuerza), W (cambiar modo AUTO/ASSIST/MANUAL)")
    print("Iniciando simulación interactiva...")
    run_interactive(sim_time=120.0)  # simula hasta 120 s o cierra ventana para parar

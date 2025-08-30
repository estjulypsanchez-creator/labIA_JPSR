import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ============================
#   MODELO DE PÉNDULO INVERTIDO
# ============================
class CartPole:
    def __init__(self, M=1.0, m=0.1, l=0.5, g=9.81, dt=0.02, force_max=10.0, x_limit=2.5):
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt
        self.force_max = force_max
        self.x_limit = x_limit
        self.reset()

    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.05  # cerca de vertical
        self.theta_dot = 0.0
        return self.state()

    def state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])

    def step(self, force):
        force = np.clip(force, -self.force_max, self.force_max)

        M, m, l, g = self.M, self.m, self.l, self.g
        x, x_dot, theta, theta_dot = self.state()

        sin_t, cos_t = np.sin(theta), np.cos(theta)
        total_mass = M + m
        temp = (force + m * l * theta_dot**2 * sin_t) / total_mass

        theta_ddot = (g * sin_t - cos_t * temp) / (l * (4/3 - (m * cos_t**2) / total_mass))
        x_ddot = temp - (m * l * theta_ddot * cos_t) / total_mass

        # integración de Euler
        x += self.dt * x_dot
        x_dot += self.dt * x_ddot
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_ddot

        # límites del carro
        if x < -self.x_limit:
            x = -self.x_limit
            x_dot = 0.0
        elif x > self.x_limit:
            x = self.x_limit
            x_dot = 0.0

        self.x, self.x_dot, self.theta, self.theta_dot = x, x_dot, theta, theta_dot
        return self.state()

# ============================
#   CONTROLADOR DIFUSO SIMPLE
# ============================
class MamdaniFuzzyController:
    def __init__(self, force_max=10.0):
        self.force_max = force_max

    def compute(self, theta, theta_dot):
        angle_deg = np.degrees(theta)

        def membership(val, low, high):
            if val <= low or val >= high:
                return 0
            else:
                return 1 - abs((val - (low+high)/2) / ((high-low)/2))

        left = membership(angle_deg, -15, -5)
        right = membership(angle_deg, 5, 15)

        force = (right - left) * self.force_max * 0.4
        return force

# ============================
#   SIMULACIÓN
# ============================
class Simulator:
    def __init__(self, env, controller):
        self.env = env
        self.controller = controller
        self.state = env.reset()
        self.keys_down = set()
        self.mode = "hybrid"  # manual + difuso

    def on_key_press(self, event):
        if event.key in ["a", "d"]:
            self.keys_down.add(event.key)

    def on_key_release(self, event):
        if event.key in ["a", "d"]:
            if event.key in self.keys_down:
                self.keys_down.remove(event.key)

    def step_manual_force(self):
        force = 0.0
        if "a" in self.keys_down:
            force -= self.env.force_max
        if "d" in self.keys_down:
            force += self.env.force_max
        return force

    def run(self, sim_time=20):
        fig, ax = plt.subplots()
        ax.set_xlim(-self.env.x_limit-0.5, self.env.x_limit+0.5)
        ax.set_ylim(-1.2, 1.2)
        ax.axvline(-self.env.x_limit, color='r', linestyle='--')
        ax.axvline(self.env.x_limit, color='r', linestyle='--')

        # --- carrito como rectángulo ---
        cart_width = 0.3
        cart_height = 0.2
        cart_patch = Rectangle((-cart_width/2, -cart_height/2),
                               cart_width, cart_height,
                               fc='black')
        ax.add_patch(cart_patch)

        pole, = ax.plot([], [], 'b-', linewidth=3)

        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        dt = self.env.dt
        frames = int(sim_time / dt)

        def init():
            cart_patch.set_xy((-cart_width/2, -cart_height/2))
            pole.set_data([], [])
            return cart_patch, pole

        def animate(i):
            manual_force = self.step_manual_force()
            fuzzy_force = self.controller.compute(self.state[2], self.state[3])

            if self.mode == "manual":
                total_force = manual_force
            elif self.mode == "fuzzy":
                total_force = fuzzy_force
            else:  # híbrido
                total_force = manual_force + fuzzy_force

            self.state = self.env.step(total_force)
            x, _, theta, _ = self.state

            # mover carrito
            cart_patch.set_x(x - cart_width/2)

            # dibujar péndulo
            pole.set_data([x, x + self.env.l * np.sin(theta)],
                          [0, self.env.l * np.cos(theta)])
            return cart_patch, pole

        ani = FuncAnimation(fig, animate, frames=frames,
                            init_func=init, blit=True, interval=dt*1000, repeat=False)
        plt.show()

if __name__ == "__main__":
    env = CartPole()
    controller = MamdaniFuzzyController()
    sim = Simulator(env, controller)
    sim.run(sim_time=30)

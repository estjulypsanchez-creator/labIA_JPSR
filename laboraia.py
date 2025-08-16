import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp
from time import time

# -------------------------
# Parámetros del sistema
# -------------------------
m1 = 0.5    # masa del carro [kg]
m2 = 0.25   # masa del péndulo [kg]
g = 9.8     # gravedad [m/s^2]
L = 0.4     # longitud total del péndulo [m]
l = L / 2   # distancia del pivote al centro de masa [m]
I = (1.0 / 12.0) * m2 * (L**2)  # inercia del péndulo alrededor de su centro
b = 0.5     # coeficiente de amortiguamiento (carrito) [N·s/m]

# -------------------------
# Parámetros de animación
# -------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, aspect='equal', xlim=(-5, 5), ylim=(-1, 1),
                     title="Péndulo Invertido July S.R. (A/D para mover, ESPACIO para reiniciar)")

# Fondo y estilo
gradient = np.linspace(0, 1, 100)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, cmap='Blues', extent=[-5, 5, -1, 1], alpha=0.3)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_facecolor('#f0f0f0')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
for s in ('top','right','bottom','left'):
    ax.spines[s].set_visible(False)
plt.title("Péndulo Invertido July S.R. (A/D para mover, ESPACIO para reiniciar)",
          fontsize=14, pad=20, fontweight='bold')

origin = [0.0, 0.0]
dt = 0.02
frames = 200
t_span = [0.0, frames * dt]

# estado temporal usado internamente por stateSpace
ss = np.zeros(4)

# límites del carrito
x_min, x_max = -4.5, 4.5

# Elementos gráficos
pendulumArm = lines.Line2D(origin, origin, color='#FF4444', linewidth=3, linestyle='-', alpha=0.8)
cart = patches.Rectangle(origin, 0.5, 0.15, color='#4488FF', alpha=0.7)

force = 0.0
force_step = 3.0  # intensidad de la fuerza aplicada al presionar A/D

# Textos de medición
vel_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
fuerza_text = ax.text(0.02, 0.70, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
angulo_text = ax.text(0.02, 0.50, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# -------------------------
# Estado y funciones dinámicas
# -------------------------
# Convención de estado: y = [x, x_dot, theta, theta_dot]
# Estado inicial por defecto (carrito en -1 m, péndulo casi invertido)
initial = np.array([-1.0, 0.0, np.pi - 0.2, 0.0])

def derivatives(t, y, F):
    """
    Ecuaciones del carro-pendulo resueltas como sistema lineal 2x2:
      M_matrix * [x_dd, theta_dd]^T = rhs
    donde y = [x, x_dot, theta, theta_dot], F = fuerza aplicada al carro
    """
    x, x_dot, theta, theta_dot = y
    S = np.sin(theta)
    C = np.cos(theta)

    # Matriz de inercia
    M11 = m1 + m2
    M12 = m2 * l * C
    M21 = m2 * l * C
    M22 = I + m2 * l**2
    Mmat = np.array([[M11, M12],
                     [M21, M22]])

    # Términos del lado derecho (resumen de fuerzas/pares)
    rhs1 = F - b * x_dot + m2 * l * (theta_dot**2) * S
    rhs2 = -m2 * g * l * S

    rhs = np.array([rhs1, rhs2])

    # resolver para [x_dd, theta_dd]
    acc = np.linalg.solve(Mmat, rhs)
    x_dd = acc[0]
    theta_dd = acc[1]

    return np.array([x_dot, x_dd, theta_dot, theta_dd])

# wrapper para solve_ivp (porque solve_ivp espera fun(t,y))
def stateSpace(t, y):
    return derivatives(t, y, force)

# -------------------------
# Eventos de teclado
# -------------------------
def reset_state():
    """Resetea el estado actual a la condición inicial."""
    global current, force
    current = initial.copy()
    force = 0.0

def on_key_press(event):
    global force
    # fuerza mientras la tecla esté presionada
    if event.key == 'a':
        force = -force_step
    elif event.key == 'd':
        force = force_step
    # aceptar tanto ' ' como 'space' por compatibilidad
    elif event.key in (' ', 'space'):
        reset_state()

def on_key_release(event):
    global force
    # al soltar A/D se anula la fuerza (asegura comportamiento intuitivo)
    if event.key in ('a', 'd'):
        force = 0.0

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# -------------------------
# Preparar integración inicial (no necesaria)
# -------------------------
# estado actual (se actualizará en cada frame)
current = initial.copy()

# -------------------------
# Animación
# -------------------------
def init():
    ax.add_patch(cart)
    ax.add_line(pendulumArm)
    vel_text.set_text('')
    fuerza_text.set_text('')
    angulo_text.set_text('')
    return pendulumArm, cart, vel_text, fuerza_text, angulo_text

def animate(i):
    global current, force

    # integrar un paso dt con solve_ivp
    sol = solve_ivp(lambda t, y: derivatives(t, y, force), [0, dt], current, t_eval=[dt], rtol=1e-8)
    current[:] = sol.y[:, -1]

    # imponer límites en la posición del carrito
    current[0] = np.clip(current[0], x_min, x_max)

    # actualizar textos
    vel_text.set_text(f'Velocidad: {current[1]:.2f} m/s')
    fuerza_text.set_text(f'Fuerza: {force:.2f} N')
    angulo_text.set_text(f'Ángulo: {np.degrees(current[2]):.1f}°')

    # coordenadas del péndulo
    xPos = current[0]
    theta = current[2]
    px = origin[0] + xPos + L * np.sin(theta)
    py = origin[1] - L * np.cos(theta)
    x_line = [origin[0] + xPos, px]
    y_line = [origin[1], py]
    pendulumArm.set_xdata(x_line)
    pendulumArm.set_ydata(y_line)

    # posición del carrito (lo dibujamos centrado en xPos)
    cartPos = [origin[0] + xPos - cart.get_width() / 2, origin[1] - cart.get_height()]
    cart.set_xy(cartPos)

    return pendulumArm, cart, vel_text, fuerza_text, angulo_text

# medir tiempo para ajustar intervalo (opcional)
t0 = time()
animate(0)
t1 = time()
interval = max(1, int(1000 * dt - (t1 - t0) * 1000))

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)
plt.show()
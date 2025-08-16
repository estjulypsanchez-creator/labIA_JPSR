from controller import Robot, DistanceSensor, Motor
import math

# Constantes del controlador
TIME_STEP = 10  # ms
MAX_SPEED = 6.28  # rad/s

# Constantes del controlador PID
KP = 20.0   # Ganancia proporcional
KI = 0.0    # Ganancia integral
KD = 2.0    # Ganancia derivativa

def main():
    # Crear instancia del robot
    robot = Robot()
    
    # Obtener referencias a los dispositivos
    left_wheel = robot.getDevice('wheel1')
    right_wheel = robot.getDevice('wheel2')
    position_sensor = robot.getDevice('position_sensor')
    
    # Habilitar el sensor de posición
    position_sensor.enable(TIME_STEP)
    
    # Variables del controlador PID
    error_integral = 0
    last_error = 0
    
    # Main loop
    while robot.step(TIME_STEP):
        # Leer el ángulo actual del péndulo
        angle = position_sensor.getValue()
        
        # Calcular el error (ángulo deseado = 0)
        error = 0 - angle
        
        # Actualizar término integral (limitado para evitar saturación)
        error_integral += error * TIME_STEP / 1000.0
        error_integral = min(max(error_integral, -1.0), 1.0)
        
        # Calcular derivada
        derivative = (error - last_error) / (TIME_STEP / 1000.0)
        
        # Calcular salida del controlador PID
        pid_output = KP * error + KI * error_integral + KD * derivative
        
        # Limitar la salida
        pid_output = min(max(pid_output, -MAX_SPEED), MAX_SPEED)
        
        # Aplicar la misma velocidad a ambas ruedas
        left_wheel.setPosition(float('inf'))
        right_wheel.setPosition(float('inf'))
        left_wheel.setVelocity(pid_output)
        right_wheel.setVelocity(pid_output)
        
        # Guardar error anterior
        last_error = error

if __name__ == "__main__":
    main()
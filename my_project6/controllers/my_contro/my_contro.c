/* my_contro.c
   Controlador para Webots:
   - A/a : avanzar (ruedas)
   - D/d : retroceder (ruedas)
   - W/w : torque manual positivo sobre eje del péndulo
   - S/s : torque manual negativo sobre eje del péndulo
   - El controlador también estabiliza el péndulo con un PID aplicado al 'pendulo_motor'.
*/

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/keyboard.h>
#include <webots/position_sensor.h>
#include <webots/console.h>
#include <stdio.h>
#include <math.h>

#define TIME_STEP 32

static double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

int main() {
  wb_robot_init();
  int ts = TIME_STEP;

  // --- Obtener dispositivos (nombres deben coincidir con .wbt) ---
  WbDeviceTag motor1 = wb_robot_get_device("motor1");
  WbDeviceTag motor2 = wb_robot_get_device("motor2");
  WbDeviceTag motor3 = wb_robot_get_device("motor3");
  WbDeviceTag motor4 = wb_robot_get_device("motor4");

  WbDeviceTag enc1 = wb_robot_get_device("encoder1");
  WbDeviceTag enc2 = wb_robot_get_device("encoder2");
  WbDeviceTag enc3 = wb_robot_get_device("encoder3");
  WbDeviceTag enc4 = wb_robot_get_device("encoder4");

  WbDeviceTag pend_sensor = wb_robot_get_device("pendulo_sensor"); // ¡usar exactamente ese nombre!
  WbDeviceTag pend_motor = wb_robot_get_device("pendulo_motor");

  // --- Comprobaciones iniciales ---
  if (!motor1 || !motor2 || !motor3 || !motor4)
    wb_console_print("ERROR: faltan uno o más motores (motor1..motor4)\n", WB_STDERR);
  if (!pend_sensor)
    wb_console_print("ERROR: no se encontró 'pendulo_sensor'\n", WB_STDERR);
  if (!pend_motor)
    wb_console_print("WARNING: no se encontró 'pendulo_motor' (no podrá aplicarse torque directo)\n", WB_STDERR);

  // --- Configurar motores de ruedas para control por velocidad ---
  if (motor1) wb_motor_set_position(motor1, INFINITY);
  if (motor2) wb_motor_set_position(motor2, INFINITY);
  if (motor3) wb_motor_set_position(motor3, INFINITY);
  if (motor4) wb_motor_set_position(motor4, INFINITY);

  if (motor1) wb_motor_set_velocity(motor1, 0.0);
  if (motor2) wb_motor_set_velocity(motor2, 0.0);
  if (motor3) wb_motor_set_velocity(motor3, 0.0);
  if (motor4) wb_motor_set_velocity(motor4, 0.0);

  // --- Habilitar encoders y sensor del péndulo ---
  if (enc1) wb_position_sensor_enable(enc1, ts);
  if (enc2) wb_position_sensor_enable(enc2, ts);
  if (enc3) wb_position_sensor_enable(enc3, ts);
  if (enc4) wb_position_sensor_enable(enc4, ts);
  if (pend_sensor) wb_position_sensor_enable(pend_sensor, ts);
  if (pend_motor) wb_motor_set_position(pend_motor, INFINITY); // torque control

  // --- Teclado ---
  wb_keyboard_enable(ts);

  // --- Parámetros de control ---
  const double WHEEL_SPEED = 4.0;       // velocidad rueda (rad/s)
  const double manual_torque_step = 0.5; // Nm aplicado con W/S
  // PID péndulo
  double Kp = 20.0;
  double Ki = 0.0;
  double Kd = 2.0;
  double integral = 0.0;
  double prev_error = 0.0;

  // límites
  const double max_pend_torque = 8.0;

  wb_console_print("Controlador iniciado. A/D: avanzar/retroceder. W/S: torque pendulo. Click en la ventana 3D.\n", WB_STDOUT);

  double manual_torque = 0.0;

  double last_time = wb_robot_get_time();
  while (wb_robot_step(ts) != -1) {
    double now = wb_robot_get_time();
    double dt = now - last_time;
    if (dt <= 0.0) dt = ts / 1000.0;

    // --- teclado (leer todas las teclas disponibles en cola) ---
    int key;
    int advance_pressed = 0, reverse_pressed = 0;
    int w_pressed = 0, s_pressed = 0;
    while ((key = wb_keyboard_get_key()) != -1) {
      if (key == 'A' || key == 'a')
        advance_pressed = 1;
      else if (key == 'D' || key == 'd')
        reverse_pressed = 1;
      else if (key == 'W' || key == 'w')
        w_pressed = 1;
      else if (key == 'S' || key == 's')
        s_pressed = 1;
    }

    // --- movimiento ruedas ---
    double v = 0.0;
    if (advance_pressed) v = WHEEL_SPEED;
    else if (reverse_pressed) v = -WHEEL_SPEED;
    else v = 0.0;

    // (si las ruedas de un lado giran invertidas, invierte la señal en dos motores)
    if (motor1) wb_motor_set_velocity(motor1, v);
    if (motor2) wb_motor_set_velocity(motor2, v);
    if (motor3) wb_motor_set_velocity(motor3, v);
    if (motor4) wb_motor_set_velocity(motor4, v);

    // --- torque manual sobre pendulo via W/S ---
    if (w_pressed) manual_torque += manual_torque_step;
    if (s_pressed) manual_torque -= manual_torque_step;
    manual_torque = clamp(manual_torque, -max_pend_torque, max_pend_torque);

    // --- PID para mantener péndulo vertical (theta = 0) ---
    double theta = 0.0;
    if (pend_sensor) theta = wb_position_sensor_get_value(pend_sensor);
    // normalizar a [-pi, pi]
    double t = fmod(theta + M_PI, 2.0*M_PI);
    if (t < 0) t += 2.0*M_PI;
    t -= M_PI;
    double error = t; // objetivo 0

    integral += error * dt;
    double derivative = (dt > 0.0) ? (error - prev_error) / dt : 0.0;
    prev_error = error;

    double u_pid = -(Kp*error + Ki*integral + Kd*derivative); // torque desde PID
    u_pid = clamp(u_pid, -max_pend_torque, max_pend_torque);

    double total_torque = clamp(u_pid + manual_torque, -max_pend_torque, max_pend_torque);

    // --- aplicar torque al eje del péndulo si existe el motor ---
    if (pend_motor) {
      wb_motor_set_torque(pend_motor, total_torque);
    }

    // --- impresión de estado cada 0.2s aprox ---
    static double acc = 0.0;
    acc += dt;
    if (acc > 0.2) {
      char buf[256];
      acc = 0.0;
    }

    last_time = now;
  }

  wb_robot_cleanup();
  return 0;
}

/*
 * REROUTE - E-puck Obstacle Avoidance Controller
 * Based on the Webots default Braitenberg controller.
 * Includes proximity sensor debugging.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include <webots/device.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/nodes.h>
#include <webots/robot.h>

#define DISTANCE_SENSORS_NUMBER 8
#define GROUND_SENSORS_NUMBER 3
#define LEDS_NUMBER 10

#define LEFT 0
#define RIGHT 1

#define MAX_SPEED 6.28
#define SENSOR_MAX_VALUE 4096.0

static WbDeviceTag distance_sensors[DISTANCE_SENSORS_NUMBER];
static double distance_sensors_values[DISTANCE_SENSORS_NUMBER];

static const char *distance_sensors_names[8] = {
    "ps0", "ps1", "ps2", "ps3",
    "ps4", "ps5", "ps6", "ps7"
};

static WbDeviceTag ground_sensors[GROUND_SENSORS_NUMBER];
static double ground_sensors_values[GROUND_SENSORS_NUMBER];

static const char *ground_sensors_names[3] = {
    "gs0", "gs1", "gs2"
};

static WbDeviceTag leds[LEDS_NUMBER];

static const char *leds_names[10] = {
    "led0", "led1", "led2", "led3", "led4",
    "led5", "led6", "led7", "led8", "led9"
};

static WbDeviceTag left_motor;
static WbDeviceTag right_motor;

static double speeds[2];

/* Original Braitenberg weights */
static const double weights[8][2] = {
    {-1.3, -1.0},
    {-1.3, -1.0},
    {-0.5,  0.5},
    { 0.0,  0.0},
    { 0.0,  0.0},
    { 0.05, -0.5},
    {-0.75, 0.0},
    {-0.75, 0.0}
};

static const double offsets[2] = {
    0.5 * MAX_SPEED,
    0.5 * MAX_SPEED
};

static int get_time_step(void) {
    static int time_step = -1;

    if (time_step == -1)
        time_step = (int)wb_robot_get_basic_time_step();

    return time_step;
}

static void step(void) {
    if (wb_robot_step(get_time_step()) == -1) {
        wb_robot_cleanup();
        exit(EXIT_SUCCESS);
    }
}

static void initialize_devices(void) {

    printf("\n--- REROUTE E-PUCK INITIALIZATION ---\n");

    for (int i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {

        distance_sensors[i] =
            wb_robot_get_device(distance_sensors_names[i]);

        if (distance_sensors[i]) {
            wb_distance_sensor_enable(
                distance_sensors[i],
                get_time_step()
            );

            printf("Detected sensor: %s\n",
                   distance_sensors_names[i]);
        } else {
            printf("WARNING: Missing sensor: %s\n",
                   distance_sensors_names[i]);
        }
    }

    for (int i = 0; i < LEDS_NUMBER; i++) {
        leds[i] = wb_robot_get_device(leds_names[i]);
    }

    for (int i = 0; i < GROUND_SENSORS_NUMBER; i++) {

        ground_sensors[i] =
            wb_robot_get_device(ground_sensors_names[i]);

        if (ground_sensors[i]) {
            wb_distance_sensor_enable(
                ground_sensors[i],
                get_time_step()
            );
        }
    }

    left_motor =
        wb_robot_get_device("left wheel motor");

    right_motor =
        wb_robot_get_device("right wheel motor");

    if (!left_motor || !right_motor) {
        fprintf(stderr, "ERROR: E-puck wheel motors not found.\n");
        wb_robot_cleanup();
        exit(EXIT_FAILURE);
    }

    wb_motor_set_position(left_motor, INFINITY);
    wb_motor_set_position(right_motor, INFINITY);

    wb_motor_set_velocity(left_motor, 0.0);
    wb_motor_set_velocity(right_motor, 0.0);

    printf("--- INITIALIZATION COMPLETE ---\n\n");

    step();
}

static void reset_actuator_values(void) {

    for (int i = 0; i < 2; i++)
        speeds[i] = 0.0;
}

static void get_sensor_input(void) {

    for (int i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {

        if (distance_sensors[i]) {

            double value =
                wb_distance_sensor_get_value(distance_sensors[i]);

            /*
             * Normalize sensor readings.
             * 0.0 = no detected obstacle.
             * 1.0 = strong proximity response.
             */

            value /= SENSOR_MAX_VALUE;

            if (value < 0.0)
                value = 0.0;

            if (value > 1.0)
                value = 1.0;

            distance_sensors_values[i] = value;

        } else {
            distance_sensors_values[i] = 0.0;
        }
    }

    for (int i = 0; i < GROUND_SENSORS_NUMBER; i++) {

        if (ground_sensors[i]) {

            ground_sensors_values[i] =
                wb_distance_sensor_get_value(ground_sensors[i]);
        }
    }
}

/* Print proximity sensor readings every 10 simulation steps. */
static void print_sensor_values(void) {

    static int counter = 0;

    counter++;

    if (counter % 10 != 0)
        return;

    printf("SENSORS: ");

    for (int i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {

        printf("ps%d=%.3f ",
               i,
               distance_sensors_values[i]);
    }

    printf("\n");
}

static bool cliff_detected(void) {

    for (int i = 0; i < GROUND_SENSORS_NUMBER; i++) {

        if (!ground_sensors[i])
            continue;

        if (ground_sensors_values[i] < 500.0)
            return true;
    }

    return false;
}

static void set_actuators(void) {

    for (int i = 0; i < LEDS_NUMBER; i++) {

        if (leds[i])
            wb_led_set(leds[i], false);
    }

    for (int i = 0; i < 2; i++) {

        if (speeds[i] > MAX_SPEED)
            speeds[i] = MAX_SPEED;

        if (speeds[i] < -MAX_SPEED)
            speeds[i] = -MAX_SPEED;
    }

    wb_motor_set_velocity(left_motor, speeds[LEFT]);
    wb_motor_set_velocity(right_motor, speeds[RIGHT]);
}

static void blink_leds(void) {

    static int counter = 0;

    counter++;

    int led_index = (counter / 10) % LEDS_NUMBER;

    if (leds[led_index])
        wb_led_set(leds[led_index], true);
}

/* Original Braitenberg obstacle-avoidance algorithm. */
static void run_braitenberg(void) {

    for (int i = 0; i < 2; i++) {

        speeds[i] = 0.0;

        for (int j = 0; j < DISTANCE_SENSORS_NUMBER; j++) {

            speeds[i] +=
                distance_sensors_values[j] * weights[j][i];
        }

        speeds[i] =
            offsets[i] + speeds[i] * MAX_SPEED;

        if (speeds[i] > MAX_SPEED)
            speeds[i] = MAX_SPEED;

        else if (speeds[i] < -MAX_SPEED)
            speeds[i] = -MAX_SPEED;
    }
}

static void go_backwards(void) {

    wb_motor_set_velocity(left_motor, -MAX_SPEED);
    wb_motor_set_velocity(right_motor, -MAX_SPEED);

    for (int i = 0; i < 5; i++)
        step();
}

static void turn_left(void) {

    wb_motor_set_velocity(left_motor, -MAX_SPEED);
    wb_motor_set_velocity(right_motor, MAX_SPEED);

    for (int i = 0; i < 5; i++)
        step();
}

int main(int argc, char **argv) {

    wb_robot_init();

    printf("\n");
    printf("====================================\n");
    printf("      REROUTE AUTONOMOUS E-PUCK     \n");
    printf("      OBSTACLE AVOIDANCE TEST       \n");
    printf("====================================\n");

    initialize_devices();

    while (true) {

        reset_actuator_values();

        get_sensor_input();

        print_sensor_values();

        blink_leds();

        if (cliff_detected()) {

            printf("WARNING: CLIFF DETECTED!\n");

            go_backwards();
            turn_left();

        } else {

            run_braitenberg();
        }

        set_actuators();

        step();
    }

    wb_robot_cleanup();

    return EXIT_SUCCESS;
}
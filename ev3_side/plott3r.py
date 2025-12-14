#!/usr/bin/env python3
from ev3dev2.motor import Motor, MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedDPS, SpeedPercent
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import TouchSensor, ColorSensor
from time import sleep
import math

class Plott3r:
    def __init__(self, config_file="config.json"):
        self.plotter_tank = MoveTank(OUTPUT_A, OUTPUT_B)
        
        self.motor_pen = Motor(OUTPUT_C)
        self.ts = TouchSensor()
        self.cs = ColorSensor()
        self.cs.mode = 'COL-COLOR'

        self.config = {
            'steps_per_mm_x': 1.7,
            'steps_per_mm_y': 1.7,
            'plotter_speed_pct': 40,
            'speed_divisor': 2.5,
            'pen_up_pos': 103,
            'pen_down_pos': -90,
            'pen_speed_sp': 150
        }

        self.plotter_tank.reset()
        self.motor_pen.reset()

        self.current_pos = [0, 0]
        self.pen_is_down = False
        
        print("Plott3r Class Initialized. Motors Reset.")

    def mm_to_steps(self, mm, axis):
        steps_per_mm = self.config['steps_per_mm_' + axis]
        steps = mm * steps_per_mm
        return int(steps)

    def move_to(self, x, y):
        """
        Movement in coordinates (x,y) with CORRECT linear interpolation (both motors work synchronously).
        """
        
        dx = x - self.current_pos[0]
        dy = y - self.current_pos[1]

        steps_x = self.mm_to_steps(dx, 'x')
        steps_y = self.mm_to_steps(dy, 'y')

        if steps_x == 0 and steps_y == 0:
            return

        abs_steps_x = abs(steps_x)
        abs_steps_y = abs(steps_y)

        base_speed = self.config['plotter_speed_pct']
        
        divisor = self.config.get('speed_divisor', 1.0)
        effective_speed = base_speed / divisor


        if abs_steps_x > abs_steps_y:
            speed_ratio = steps_y / steps_x 
            speed_x = SpeedPercent(effective_speed * math.copysign(1, steps_x))
            speed_y = SpeedPercent(effective_speed * speed_ratio * math.copysign(1, steps_x))
            total_degrees = abs_steps_x
        
        elif abs_steps_y > abs_steps_x:
            speed_ratio = steps_x / steps_y
            speed_y = SpeedPercent(effective_speed * math.copysign(1, steps_y))
            speed_x = SpeedPercent(effective_speed * speed_ratio * math.copysign(1, steps_y))
            total_degrees = abs_steps_y
            
        else:
            speed_x = SpeedPercent(effective_speed * math.copysign(1, steps_x))
            speed_y = SpeedPercent(effective_speed * math.copysign(1, steps_y))
            total_degrees = abs_steps_x

        self.plotter_tank.on_for_degrees(
            speed_x,
            speed_y,
            total_degrees,
            brake=True,
            block=True
        )

        self.current_pos = [x, y]


    def pen_up(self):
        """Raise the pen"""
        if self.pen_is_down:
            print('Pen Up')
            self.motor_pen.stop()
            
            pen_speed = SpeedDPS(self.config['pen_speed_sp'])
            degrees_to_move = self.config['pen_up_pos']
            
            self.motor_pen.on_for_degrees(pen_speed, degrees_to_move, block=True)
            
            self.pen_is_down = False
            sleep(0.1)


    def pen_down(self):
        """Lower the pen"""
        if not self.pen_is_down:
            print('Pen Down')
            self.motor_pen.stop() 
            
            pen_speed_val = self.config['pen_speed_sp']
            degrees_to_move = self.config['pen_down_pos']
            
            pen_speed = SpeedDPS(pen_speed_val * -1)
            degrees_abs = abs(degrees_to_move)

            self.motor_pen.on_for_degrees(pen_speed, degrees_abs, block=True)

            self.pen_is_down = True
            sleep(0.1)


    def calibrate(self):
        """Calibration"""
        print('Calibrate')
        
        self.pen_is_down = True
        self.pen_up()
        self.motor_pen.on_for_seconds(speed=50, seconds=1)


        cal_speed_x = SpeedPercent(25) 
        cal_speed_y = SpeedPercent(25)

        self.plotter_tank.on(cal_speed_x, SpeedPercent(0))
        
        while True:
            if self.ts.is_pressed:
                self.plotter_tank.stop()
                break
            sleep(0.01)
        
        self.plotter_tank.on_for_degrees(SpeedPercent(-25), SpeedPercent(0), 1520, block=True)
        
        print("Insert paper")
        
        while True:
            if self.cs.color == 6:
                self.plotter_tank.on(SpeedPercent(0), cal_speed_y)
                break
            sleep(0.01)
        
        while True:
            if self.cs.color == 0:
                self.plotter_tank.stop()
                break
            sleep(0.01)
        
        self.plotter_tank.on_for_degrees(SpeedPercent(0), SpeedPercent(-25), 1800, block=True)
        
        self.current_pos = [0, 0]
        self.plotter_tank.reset()
        print("Calibration complete")

    def draw_line(self, x1, y1, x2, y2):
        self.move_to(x1, y1)
        self.pen_down()
        self.move_to(x2, y2)
        self.pen_up()
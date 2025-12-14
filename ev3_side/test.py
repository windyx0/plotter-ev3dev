#!/usr/bin/env python3
from ev3dev2.motor import *

y_motor = Motor(OUTPUT_B)

y_motor.on_for_degrees(SpeedPercent(25), 500, block=True)
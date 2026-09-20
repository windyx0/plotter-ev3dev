#!/usr/bin/env python3
import ast
import json
import math
import os
from time import sleep

from ev3dev2.motor import Motor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedDPS, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3
from ev3dev2.sensor.lego import TouchSensor, ColorSensor


class Plott3r:
    REQUIRED_CONFIG_KEYS = (
        "steps_per_mm_x",
        "steps_per_mm_y",
        "pen_up_pos",
        "pen_down_pos",
        "pen_speed_sp",
        "arc_segment_mm",
        "arc_max_segments",
    )

    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.motor_x = Motor(OUTPUT_A)
        self.motor_y = Motor(OUTPUT_B)
        self.motor_x2 = None
        if bool(self.config.get("dual_x_motor", True)):
            try:
                self.motor_x2 = Motor(OUTPUT_D)
                print("Dual X motor enabled on OUTPUT_D")
            except Exception as exc:
                raise RuntimeError("Dual X motor requested, but OUTPUT_D is not available: {}".format(exc))
        self.motor_pen = Motor(OUTPUT_C)
        self.ts = TouchSensor(address=self.config.get("touch_right_port", INPUT_2))
        self.ts_x2 = None
        if self.motor_x2 is not None:
            try:
                self.ts_x2 = TouchSensor(address=self.config.get("touch_left_port", INPUT_3))
                print("Left X touch sensor enabled")
            except Exception as exc:
                print("Left X touch sensor disabled: {}".format(exc))
        self.cs = ColorSensor(address=self.config.get("color_sensor_port", INPUT_1))
        self.cs.mode = "COL-COLOR"

        self.motor_x.reset()
        self.motor_y.reset()
        if self.motor_x2 is not None:
            self.motor_x2.reset()
        self.motor_pen.reset()

        self.current_pos = [0.0, 0.0]
        self.current_steps = {"x": 0, "y": 0}
        self.last_motion_speed_pct = 0.0
        self.pen_is_down = False
        print("Plott3r initialized. Motors reset.")

    def _load_config(self, config_file):
        config_path = config_file
        if not os.path.isabs(config_file):
            config_path = os.path.join(os.path.dirname(__file__), config_file)

        if not os.path.exists(config_path):
            raise RuntimeError("Config file not found: {}".format(config_path))

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                loaded = ast.literal_eval(raw)
            if not isinstance(loaded, dict):
                raise ValueError("Config root must be a JSON object")

            missing = [k for k in self.REQUIRED_CONFIG_KEYS if k not in loaded]
            if missing:
                raise ValueError("Missing required config keys: {}".format(", ".join(missing)))

            # New unified speed control: 0.0..1.0 where 0.5 == 50% speed.
            # Backward compatible with legacy plotter_speed_pct/speed_divisor.
            if "speed" not in loaded:
                legacy_pct = float(loaded.get("plotter_speed_pct", 50.0))
                legacy_div = max(0.1, float(loaded.get("speed_divisor", 1.0)))
                loaded["speed"] = max(0.05, min(1.0, (legacy_pct / legacy_div) / 100.0))

            print("Loaded config: {}".format(config_path))
            return loaded
        except Exception as exc:
            raise RuntimeError("Failed to load config '{}': {}".format(config_path, exc))

    def mm_to_steps(self, mm, axis):
        return int(mm * float(self.config["steps_per_mm_" + axis]))

    def _mm_to_abs_steps(self, mm, axis):
        return int(round(float(mm) * float(self.config["steps_per_mm_" + axis])))

    def _steps_to_mm(self, steps, axis):
        return float(steps) / float(self.config["steps_per_mm_" + axis])

    def set_logical_position(self, x, y):
        self.current_steps = {
            "x": self._mm_to_abs_steps(float(x), "x"),
            "y": self._mm_to_abs_steps(float(y), "y"),
        }
        self.current_pos = [
            self._steps_to_mm(self.current_steps["x"], "x"),
            self._steps_to_mm(self.current_steps["y"], "y"),
        ]
        print("Logical position set to X={} Y={} without moving".format(self.current_pos[0], self.current_pos[1]))

    def _speed_percent(self, value):
        return SpeedPercent(max(-100, min(100, value)))

    def _x2_speed(self, raw_speed_x):
        if bool(self.config.get("dual_x_motor_inverted", False)):
            return -raw_speed_x
        return raw_speed_x

    def _active_motion_motors(self):
        motors = [self.motor_x, self.motor_y]
        if self.motor_x2 is not None:
            motors.append(self.motor_x2)
        return motors

    def _wait_motion_done(self, motors):
        for motor in motors:
            try:
                motor.wait_until_not_moving()
            except Exception:
                pass

    def _run_x_motors_for_degrees(self, raw_speed_x, degrees, brake=True):
        degrees_abs = abs(int(degrees))
        if degrees_abs == 0:
            return

        active = []
        self.motor_x.on_for_degrees(self._speed_percent(raw_speed_x), degrees_abs, brake=brake, block=False)
        active.append(self.motor_x)
        if self.motor_x2 is not None:
            self.motor_x2.on_for_degrees(
                self._speed_percent(self._x2_speed(raw_speed_x)), degrees_abs, brake=brake, block=False
            )
            active.append(self.motor_x2)
        elif bool(self.config.get("dual_x_motor", True)):
            raise RuntimeError("Dual X motor is enabled, but OUTPUT_D motor is missing")

        self._wait_motion_done(active)

    def _effective_speed_pct(self) -> float:
        raw_speed = self.config.get("speed", 0.5)
        try:
            speed = float(raw_speed)
        except Exception:
            speed = 0.5
        speed = max(0.05, min(1.0, speed))
        return speed * 100.0

    def _segment_speed_pct(self, abs_steps_x, abs_steps_y, speed_scale=1.0):
        base_speed = self._effective_speed_pct()
        seg_steps = max(abs_steps_x, abs_steps_y)
        full_speed_steps = max(1.0, float(self.config.get("short_move_full_steps", 120.0)))
        # Slow down short moves to reduce overshoot / skipped steps on corners and tiny segments.
        short_factor = 0.25 + 0.75 * min(1.0, seg_steps / full_speed_steps)
        min_speed_pct = max(5.0, float(self.config.get("min_speed_pct", 12.0)))
        scaled = base_speed * short_factor * max(0.1, float(speed_scale))
        target = max(min_speed_pct, min(100.0, scaled))
        accel_limit = max(1.0, float(self.config.get("max_speed_step_pct", 4.0)))
        if self.last_motion_speed_pct > 0:
            target = min(target, self.last_motion_speed_pct + accel_limit)
        self.last_motion_speed_pct = target
        return target

    def move_to(self, x, y, speed_scale=1.0):
        target_steps_x = self._mm_to_abs_steps(x, "x")
        target_steps_y = self._mm_to_abs_steps(y, "y")
        steps_x = target_steps_x - self.current_steps["x"]
        steps_y = target_steps_y - self.current_steps["y"]
        if steps_x == 0 and steps_y == 0:
            self.current_pos = [
                self._steps_to_mm(self.current_steps["x"], "x"),
                self._steps_to_mm(self.current_steps["y"], "y"),
            ]
            return

        abs_steps_x = abs(steps_x)
        abs_steps_y = abs(steps_y)
        effective_speed = self._segment_speed_pct(abs_steps_x, abs_steps_y, speed_scale=speed_scale)

        raw_speed_x = 0.0
        raw_speed_y = 0.0
        if abs_steps_x > abs_steps_y:
            speed_ratio = steps_y / steps_x
            raw_speed_x = effective_speed * math.copysign(1, steps_x)
            raw_speed_y = effective_speed * speed_ratio * math.copysign(1, steps_x)
            total_degrees = abs_steps_x
        elif abs_steps_y > abs_steps_x:
            speed_ratio = steps_x / steps_y
            raw_speed_y = effective_speed * math.copysign(1, steps_y)
            raw_speed_x = effective_speed * speed_ratio * math.copysign(1, steps_y)
            total_degrees = abs_steps_y
        else:
            raw_speed_x = effective_speed * math.copysign(1, steps_x)
            raw_speed_y = effective_speed * math.copysign(1, steps_y)
            total_degrees = abs_steps_x

        # Keep diagonal moves inside the same power budget as axial moves.
        # For 45deg this applies ~1/sqrt(2) scaling automatically.
        if bool(self.config.get("limit_vector_power", True)):
            vector_speed = math.hypot(raw_speed_x, raw_speed_y)
            max_vector_speed = float(self.config.get("max_vector_speed_pct", effective_speed))
            max_vector_speed = max(1.0, min(100.0, max_vector_speed))
            if vector_speed > max_vector_speed and vector_speed > 1e-6:
                k = max_vector_speed / vector_speed
                raw_speed_x *= k
                raw_speed_y *= k

        active = []
        if abs_steps_x > 0:
            self.motor_x.on_for_degrees(self._speed_percent(raw_speed_x), abs_steps_x, brake=True, block=False)
            active.append(self.motor_x)
            if self.motor_x2 is not None:
                self.motor_x2.on_for_degrees(
                    self._speed_percent(self._x2_speed(raw_speed_x)), abs_steps_x, brake=True, block=False
                )
                active.append(self.motor_x2)
            elif bool(self.config.get("dual_x_motor", True)):
                raise RuntimeError("Dual X motor is enabled, but OUTPUT_D motor is missing")
        if abs_steps_y > 0:
            self.motor_y.on_for_degrees(self._speed_percent(raw_speed_y), abs_steps_y, brake=True, block=False)
            active.append(self.motor_y)

        self._wait_motion_done(active)
        settle = float(self.config.get("move_settle_sec", 0.0))
        if settle > 0:
            sleep(settle)
        self.current_steps = {"x": target_steps_x, "y": target_steps_y}
        self.current_pos = [
            self._steps_to_mm(self.current_steps["x"], "x"),
            self._steps_to_mm(self.current_steps["y"], "y"),
        ]

    def draw_arc(self, x_end, y_end, i_offset, j_offset, clockwise=True):
        start_x, start_y = self.current_pos
        center_x = start_x + float(i_offset)
        center_y = start_y + float(j_offset)

        radius = math.hypot(start_x - center_x, start_y - center_y)
        if radius < 1e-6:
            self.move_to(x_end, y_end)
            return

        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        end_angle = math.atan2(float(y_end) - center_y, float(x_end) - center_x)

        sweep = end_angle - start_angle
        if clockwise and sweep >= 0:
            sweep -= 2 * math.pi
        elif not clockwise and sweep <= 0:
            sweep += 2 * math.pi

        arc_length = abs(sweep) * radius
        segment_len = max(0.2, float(self.config.get("arc_segment_mm", 1.2)))
        max_segments = max(8, int(self.config.get("arc_max_segments", 240)))
        segments = max(1, min(max_segments, int(math.ceil(arc_length / segment_len))))

        for idx in range(1, segments + 1):
            t = idx / float(segments)
            angle = start_angle + sweep * t
            px = center_x + radius * math.cos(angle)
            py = center_y + radius * math.sin(angle)
            arc_speed_factor = max(0.2, float(self.config.get("arc_speed_factor", 0.72)))
            self.move_to(px, py, speed_scale=arc_speed_factor)

        # Ensure exact final point.
        self.move_to(float(x_end), float(y_end), speed_scale=max(0.2, float(self.config.get("arc_speed_factor", 0.72))))

    def pen_up(self):
        if not self.pen_is_down:
            return

        print("Pen Up")
        self.motor_pen.stop()
        pen_speed = SpeedDPS(abs(int(self.config["pen_speed_sp"])))
        degrees_to_move = abs(int(self.config["pen_up_pos"]))
        self.motor_pen.on_for_degrees(pen_speed, degrees_to_move, block=True)
        self.pen_is_down = False
        self.last_motion_speed_pct = 0.0
        sleep(max(0.0, float(self.config.get("pen_settle_sec", 0.02))))

    def pen_down(self):
        if self.pen_is_down:
            return

        print("Pen Down")
        self.motor_pen.stop()
        pen_speed = SpeedDPS(-abs(int(self.config["pen_speed_sp"])))
        degrees_abs = abs(int(self.config["pen_down_pos"]))
        self.motor_pen.on_for_degrees(pen_speed, degrees_abs, block=True)
        self.pen_is_down = True
        self.last_motion_speed_pct = 0.0
        sleep(max(0.0, float(self.config.get("pen_settle_sec", 0.02))))

    def _touch_pressed(self, sensor):
        if sensor is None:
            return False
        try:
            return bool(sensor.is_pressed)
        except Exception:
            return False

    def _home_x_near_touch(self):
        speed_mag = abs(float(self.config.get("x_home_speed_pct", 18.0)))
        home_direction = -1.0 if float(self.config.get("x_home_direction", -1.0)) < 0 else 1.0
        speed = speed_mag * home_direction
        backoff_degrees = abs(int(self.config.get("x_home_backoff_degrees", 50)))
        backoff_direction = -1.0 if float(self.config.get("x_backoff_direction", 1.0)) < 0 else 1.0
        timeout_sec = float(self.config.get("x_home_timeout_sec", 18.0))
        tick = 0.01
        elapsed = 0.0

        motors = [{"motor": self.motor_x, "speed": speed}]
        if self.motor_x2 is not None:
            motors.append(
                {
                    "motor": self.motor_x2,
                    "speed": self._x2_speed(speed),
                }
            )

        print(
            "X home: direction={} speed={} touch_right={} touch_left={} dual_x={}".format(
                home_direction,
                speed,
                self.config.get("touch_right_port", "in2"),
                self.config.get("touch_left_port", "in3"),
                self.motor_x2 is not None,
            )
        )
        for item in motors:
            item["motor"].on(self._speed_percent(item["speed"]))

        touched = False
        while elapsed < timeout_sec:
            if self._touch_pressed(self.ts) or self._touch_pressed(self.ts_x2):
                touched = True
                break
            sleep(tick)
            elapsed += tick

        for item in motors:
            item["motor"].stop()
        if not touched:
            raise RuntimeError("X home touch sensor timeout")

        backoff_scale = max(0.1, min(1.0, float(self.config.get("x_backoff_speed_scale", 0.6))))
        backoff_speed = speed_mag * backoff_scale * backoff_direction
        print("X home backoff: direction={} degrees={} speed={}".format(backoff_direction, backoff_degrees, backoff_speed))
        self._run_x_motors_for_degrees(backoff_speed, backoff_degrees, brake=True)

    def calibrate(self):
        print("Calibrate")
        self.pen_is_down = True
        self.pen_up()

        self.motor_pen.on_for_seconds(speed=50, seconds=1)

        self.calibrate_x(reset_coords=False)
        self.calibrate_paper(reset_coords=False)

        self.current_pos = [0.0, 0.0]
        self.current_steps = {"x": 0, "y": 0}
        self.motor_x.reset()
        self.motor_y.reset()
        if self.motor_x2 is not None:
            self.motor_x2.reset()

    def calibrate_x(self, reset_coords=True):
        print("Calibrate X")
        self._home_x_near_touch()
        if reset_coords:
            self.current_pos[0] = 0.0
            self.current_steps["x"] = 0
            self.motor_x.reset()
            if self.motor_x2 is not None:
                self.motor_x2.reset()

    def calibrate_paper(self, reset_coords=True):
        print("Insert paper")
        while True:
            if self.cs.color == 6:
                self.motor_y.on(SpeedPercent(25))
                break
            sleep(0.01)

        while True:
            if self.cs.color == 0:
                self.motor_y.stop()
                break
            sleep(0.01)

        self.motor_y.on_for_degrees(SpeedPercent(-25), 1800, block=True)
        if reset_coords:
            self.current_pos[1] = 0.0
            self.current_steps["y"] = 0
            self.motor_y.reset()
        print("Calibration complete")

    def draw_line(self, x1, y1, x2, y2):
        self.move_to(x1, y1)
        self.pen_down()
        self.move_to(x2, y2)
        self.pen_up()

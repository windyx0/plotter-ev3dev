#!/usr/bin/env python3
import os
import time
import json
import socket
import hashlib
from ev3dev2.display import Display
from ev3dev2.button import Button
from ev3dev2.sound import Sound
from ev3dev2.motor import SpeedPercent
import ev3dev2.fonts as fonts

# Проверка наличия PIL
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL not found. Run: sudo apt-get install python3-pil")
    raise

from plott3r import Plott3r

PROGRESS_FILE = "/home/robot/ev3_side/print_progress.json"
CACHE_FILE = "/home/robot/ev3_side/gcode_cache.json"
LOG_FILE = "/home/robot/ev3_side/main_menu.log"


class PlotterGUI:
    def __init__(self):
        self.lcd = Display()
        self.btn = Button()
        self.sound = Sound()
        self.script_dir = os.path.dirname(__file__)
        self.progress_paths = [
            PROGRESS_FILE,
            os.path.join(self.script_dir, "print_progress.json"),
            os.path.join(os.getcwd(), "print_progress.json"),
        ]
        self.cache_paths = [
            CACHE_FILE,
            os.path.join(self.script_dir, "gcode_cache.json"),
            os.path.join(os.getcwd(), "gcode_cache.json"),
            "/home/robot/gcode_cache.json",
            "/home/robot/print_files/gcode_cache.json",
            "/home/robot/lms2012/prjs/gcode_cache.json",
        ]
        self.cache_backup_gcode_paths = [
            "/home/robot/ev3_side/last_job.gcode",
            os.path.join(self.script_dir, "last_job.gcode"),
            os.path.join(os.getcwd(), "last_job.gcode"),
            "/home/robot/print_files/last_job.gcode",
        ]
        self.log_paths = [
            LOG_FILE,
            os.path.join(self.script_dir, "main_menu.log"),
            os.path.join(os.getcwd(), "main_menu.log"),
        ]
        self.active_commands_hash = None
        self.active_commands_count = None

        try:
            self.font_header = fonts.load('helvB14')
            self.font_item = fonts.load('helvR12')
        except Exception:
            self.font_header = None
            self.font_item = None

        self.files_dir = "/home/robot/print_files"
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

        self.draw_splash("Starting...")
        try:
            self.plotter = Plott3r()
        except Exception as e:
            self.show_error("Motor Error:\n{}".format(e))
            self.plotter = None

        self.cached_ip = self.get_local_ip()
        self.log("PlotterGUI started. cwd={}".format(os.getcwd()))

    def log(self, msg):
        line = "[{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        for p in self.log_paths:
            try:
                base = os.path.dirname(p)
                if base and not os.path.exists(base):
                    os.makedirs(base)
                with open(p, "a") as f:
                    f.write(line)
                break
            except Exception:
                continue

    def _read_json_first_existing(self, paths):
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        return json.load(f)
                except Exception:
                    continue
        return None

    def _read_json_all_existing(self, paths):
        found = []
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        payload = json.load(f)
                    found.append((p, payload))
                except Exception:
                    continue
        return found

    def _write_json_all(self, paths, payload):
        for p in paths:
            try:
                base = os.path.dirname(p)
                if base and not os.path.exists(base):
                    os.makedirs(base)
                with open(p, "w") as f:
                    json.dump(payload, f)
            except Exception:
                pass

    def _extract_commands_from_payload(self, payload):
        if isinstance(payload, list):
            return payload if payload and isinstance(payload[0], str) else []
        if not isinstance(payload, dict):
            return []
        for key in ("commands", "gcode_lines", "lines"):
            value = payload.get(key)
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value
        return []

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
        except Exception:
            return "No WiFi"
        finally:
            s.close()

    def draw_splash(self, text):
        self.lcd.clear()
        self.lcd.draw.text((10, 50), text, font=self.font_header)
        self.lcd.update()

    def wait_any_key(self):
        while self.btn.any():
            time.sleep(0.1)
        while not self.btn.any():
            time.sleep(0.1)
        while self.btn.any():
            time.sleep(0.1)

    def show_error(self, text):
        self.lcd.clear()
        self.lcd.draw.text((0, 0), "INFO / ERROR", font=self.font_header)
        self.lcd.draw.text((0, 30), str(text), font=self.font_item)
        self.lcd.draw.text((0, 100), "Press any key...", font=self.font_item)
        self.lcd.update()
        self.sound.beep()
        self.wait_any_key()

    def load_progress(self):
        data = self._read_json_first_existing(self.progress_paths)
        if data is None:
            self.log("load_progress: no file found in paths={}".format(self.progress_paths))
            return {"progress_line": 0, "pen_is_down": False}
        try:
            self.log("load_progress: loaded progress_line={} pen_is_down={}".format(
                int(data.get("progress_line", 0)),
                bool(data.get("pen_is_down", False))
            ))
            return {
                "progress_line": int(data.get("progress_line", 0)),
                "pen_is_down": bool(data.get("pen_is_down", False)),
                "commands_count": int(data.get("commands_count", 0) or 0),
                "commands_hash": data.get("commands_hash")
            }
        except Exception:
            self.log("load_progress: invalid payload={}".format(data))
            return {"progress_line": 0, "pen_is_down": False}

    def save_progress(self, line_number, pen_is_down):
        payload = {"progress_line": int(line_number), "pen_is_down": bool(pen_is_down)}
        if self.active_commands_count is not None:
            payload["commands_count"] = int(self.active_commands_count)
        if self.active_commands_hash:
            payload["commands_hash"] = self.active_commands_hash
        self._write_json_all(self.progress_paths, payload)
        self.log("save_progress: line={} pen_down={}".format(int(line_number), bool(pen_is_down)))

    def commands_hash(self, commands):
        h = hashlib.sha1()
        for cmd in commands:
            h.update(str(cmd).strip().encode("utf-8", "ignore"))
            h.update(b"\n")
        return h.hexdigest()

    def load_gcode_cache(self):
        best_commands = None
        best_path = None
        best_mtime = -1.0

        # 1) Check known paths first.
        for p in self.cache_paths:
            self.log("cache check known exists? path={} -> {}".format(p, os.path.exists(p)))
        for p, payload in self._read_json_all_existing(self.cache_paths):
            try:
                commands = self._extract_commands_from_payload(payload)
                clen = len(commands) if isinstance(commands, list) else 0
                self.log("cache candidate known: path={} commands_len={}".format(p, clen))
                if commands:
                    mtime = os.path.getmtime(p)
                    if mtime > best_mtime:
                        best_mtime = mtime
                        best_commands = commands
                        best_path = p
            except Exception:
                pass

        # 2) Fallback: scan common EV3 folders for any gcode_cache.json.
        if best_commands is None:
            search_roots = ["/home/robot", self.script_dir, os.getcwd()]
            for root in search_roots:
                if not root or not os.path.exists(root):
                    continue
                try:
                    for current_root, _, files in os.walk(root):
                        if "gcode_cache.json" not in files:
                            continue
                        p = os.path.join(current_root, "gcode_cache.json")
                        try:
                            with open(p, "r") as f:
                                payload = json.load(f)
                            commands = self._extract_commands_from_payload(payload)
                            clen = len(commands) if isinstance(commands, list) else 0
                            self.log("cache candidate scan: path={} commands_len={}".format(p, clen))
                            if commands:
                                mtime = os.path.getmtime(p)
                                if mtime > best_mtime:
                                    best_mtime = mtime
                                    best_commands = commands
                                    best_path = p
                        except Exception:
                            continue
                except Exception:
                    continue

        # 3) Fallback: plain-text backup gcode files.
        if best_commands is None:
            for p in self.cache_backup_gcode_paths:
                self.log("cache check backup exists? path={} -> {}".format(p, os.path.exists(p)))
                if not os.path.exists(p):
                    continue
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        commands = self.parse_gcode_text(f.read())
                    self.log("cache candidate backup: path={} commands_len={}".format(p, len(commands)))
                    if commands:
                        mtime = os.path.getmtime(p)
                        if mtime > best_mtime:
                            best_mtime = mtime
                            best_commands = commands
                            best_path = p
                except Exception:
                    continue

        if best_commands is None:
            self.log("load_gcode_cache: NOT FOUND")
        else:
            self.log("load_gcode_cache: selected path={} commands={}".format(best_path, len(best_commands)))
        return best_commands

    def save_gcode_cache(self, commands):
        self._write_json_all(self.cache_paths, {"commands": commands})
        for p in self.cache_backup_gcode_paths:
            try:
                base = os.path.dirname(p)
                if base and not os.path.exists(base):
                    os.makedirs(base)
                with open(p, "w", encoding="utf-8") as f:
                    f.write("\n".join(commands) + "\n")
                self.log("save_gcode_cache backup: path={} commands={}".format(p, len(commands) if isinstance(commands, list) else -1))
            except Exception as exc:
                self.log("save_gcode_cache backup fail: path={} err={}".format(p, exc))
        self.log("save_gcode_cache: commands={}".format(len(commands) if isinstance(commands, list) else -1))

    def parse_motion_values(self, cmd, current_x, current_y):
        x = current_x
        y = current_y
        i_val = 0.0
        j_val = 0.0
        for part in cmd.split():
            if part.startswith('X'):
                x = float(part[1:])
            elif part.startswith('Y'):
                y = float(part[1:])
            elif part.startswith('I'):
                i_val = float(part[1:])
            elif part.startswith('J'):
                j_val = float(part[1:])
        return x, y, i_val, j_val

    def find_last_position(self, commands, start_line):
        x = 0.0
        y = 0.0
        for idx in range(0, min(start_line, len(commands))):
            cmd = commands[idx].strip().upper()
            if cmd.startswith(("G0", "G1", "G2", "G3")):
                x, y, _, _ = self.parse_motion_values(cmd, x, y)
        return x, y

    def parse_gcode_text(self, text):
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(';'):
                continue
            if ';' in line:
                line = line.split(';', 1)[0].strip()
            if line:
                lines.append(line)
        return lines

    def load_commands_from_file(self, filepath):
        if filepath.lower().endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data.get('commands', [])

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return self.parse_gcode_text(f.read())

    def menu_selector(self, title, items):
        current_index = 0
        while True:
            xres = int(getattr(self.lcd, "xres", 178))
            yres = int(getattr(self.lcd, "yres", 128))
            header_h = 18
            row_h = 16
            rows = max(5, int((yres - (header_h + 2)) // row_h))
            self.lcd.clear()
            self.lcd.draw.rectangle((0, 0, xres - 1, header_h), fill='black')
            self.lcd.draw.text((5, 2), title, fill='white', font=self.font_item)
            self.lcd.draw.text((max(80, xres - 72), 2), self.cached_ip[:10], fill='white', font=self.font_item)

            start_idx = 0
            if current_index >= rows:
                start_idx = current_index - rows + 1

            y = header_h + 2
            for i in range(start_idx, min(start_idx + rows, len(items))):
                prefix = " "
                if i == current_index:
                    prefix = ">"
                    self.lcd.draw.rectangle((0, y, xres - 1, y + row_h - 1), fill='black')
                    color = 'white'
                else:
                    color = 'black'

                text = "{} {}".format(prefix, items[i])
                self.lcd.draw.text((2, y), text, font=self.font_item, fill=color)
                y += row_h

            if start_idx > 0:
                self.lcd.draw.text((xres - 10, header_h + 2), "^", font=self.font_item)
            if start_idx + rows < len(items):
                self.lcd.draw.text((xres - 10, yres - 16), "v", font=self.font_item)

            self.lcd.update()

            if self.btn.down:
                current_index = (current_index + 1) % len(items)
                time.sleep(0.2)
            elif self.btn.up:
                current_index = (current_index - 1) % len(items)
                time.sleep(0.2)
            elif self.btn.enter:
                time.sleep(0.2)
                return current_index
            elif self.btn.backspace or self.btn.left or self.btn.right:
                while self.btn.backspace or self.btn.left or self.btn.right:
                    time.sleep(0.1)
                return -1

    def run_server_mode(self):
        ip_addr = self.get_local_ip()
        self.cached_ip = ip_addr

        self.lcd.clear()
        self.lcd.draw.text((5, 5), "SERVER MODE", font=self.font_header)
        self.lcd.draw.text((5, 30), "IP: {}".format(ip_addr), font=self.font_item)
        self.lcd.draw.text((5, 50), "Port: 15614", font=self.font_item)
        self.lcd.draw.text((5, 80), "Waiting...", font=self.font_header)
        self.lcd.draw.text((5, 110), "[L/R] to Exit", font=self.font_item)
        self.lcd.update()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.settimeout(0.5)
        try:
            server_socket.bind(('0.0.0.0', 15614))
            server_socket.listen(1)
        except Exception as e:
            self.show_error(str(e))
            return

        while True:
            if self.btn.backspace or self.btn.left or self.btn.right:
                while self.btn.backspace or self.btn.left or self.btn.right:
                    time.sleep(0.1)
                break

            try:
                conn, addr = server_socket.accept()
                self.draw_splash("Connected:\n{}".format(addr[0]))

                data_acc = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data_acc += chunk

                try:
                    json_data = json.loads(data_acc.decode('utf-8'))
                    commands = json_data.get('commands', [])
                    # Immediate ACK to PC so sender doesn't wait on timeout.
                    try:
                        conn.sendall(b"OK")
                    except Exception:
                        pass
                    self.save_gcode_cache(commands)
                    self.save_progress(0, False)
                    conn.close()
                    self.execute_gcode(commands, start_line=0, start_pen_is_down=False, save_resume=True)
                    
                    self.lcd.clear()
                    self.lcd.draw.text((5, 5), "SERVER MODE", font=self.font_header)
                    self.lcd.draw.text((5, 30), "IP: {}".format(ip_addr), font=self.font_item)
                    self.lcd.draw.text((5, 50), "Port: 15614", font=self.font_item)
                    self.lcd.draw.text((5, 80), "Waiting...", font=self.font_header)
                    self.lcd.draw.text((5, 110), "[L/R] to Exit", font=self.font_item)
                    self.lcd.update()
                    continue
                except ValueError:
                    pass

                conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                self.show_error(str(e))

        server_socket.close()

    def run_pen_test(self):
        self.lcd.clear()
        self.lcd.draw.text((5, 0), "PEN TEST", font=self.font_header)
        self.lcd.draw.text((5, 30), "Enter: Toggle", font=self.font_item)
        self.lcd.draw.text((5, 50), "L/R: Exit", font=self.font_item)
        self.lcd.update()

        while True:
            if self.btn.enter:
                if self.plotter.pen_is_down:
                    self.plotter.pen_up()
                else:
                    self.plotter.pen_down()
                time.sleep(0.2)
                while self.btn.enter:
                    time.sleep(0.05)
            if self.btn.backspace or self.btn.left or self.btn.right:
                if self.plotter.pen_is_down:
                    self.plotter.pen_up()
                while self.btn.backspace or self.btn.left or self.btn.right:
                    time.sleep(0.1)
                break
            time.sleep(0.05)

    def run_pen_calibrate(self):
        self.lcd.clear()
        self.lcd.draw.text((5, 0), "PEN CALIBRATE", font=self.font_header)
        self.lcd.draw.text((5, 30), "Up/Dn: Move pen", font=self.font_item)
        self.lcd.draw.text((5, 50), "L/R: Exit", font=self.font_item)
        self.lcd.update()

        while True:
            if self.btn.up:
                self.plotter.motor_pen.on_for_degrees(SpeedPercent(10), 10, block=True)
            elif self.btn.down:
                self.plotter.motor_pen.on_for_degrees(SpeedPercent(-10), 10, block=True)
            elif self.btn.left or self.btn.right:
                while self.btn.left or self.btn.right:
                    time.sleep(0.1)
                break
            time.sleep(0.05)

    def run_file_browser(self):
        files = [
            f for f in os.listdir(self.files_dir)
            if f.lower().endswith(('.json', '.gcode', '.nc', '.txt'))
        ]
        if not files:
            self.show_error("No print files\nin print_files")
            return

        files.sort()
        selected = self.menu_selector("Select File", files)
        if selected != -1:
            filename = files[selected]
            filepath = os.path.join(self.files_dir, filename)

            conf = self.menu_selector("Print {}?".format(filename), ["Yes", "No"])
            if conf == 0:
                try:
                    commands = self.load_commands_from_file(filepath)
                    if not commands:
                        self.show_error("Empty file:\n{}".format(filename))
                        return
                    self.save_gcode_cache(commands)
                    self.save_progress(0, False)
                    self.execute_gcode(commands, start_line=0, start_pen_is_down=False, save_resume=True)
                except Exception as e:
                    self.show_error("File Error:\n{}".format(e))

    def continue_unfinished(self):
        progress_data = self.load_progress()
        start_line = progress_data.get("progress_line", 0)
        start_pen = progress_data.get("pen_is_down", False)
        progress_count = int(progress_data.get("commands_count", 0) or 0)
        progress_hash = progress_data.get("commands_hash")
        self.log("continue_unfinished: progress_line={} pen={}".format(start_line, start_pen))

        if start_line <= 0:
            self.show_error("No unfinished\nprint found")
            return

        commands = self.load_gcode_cache()
        if not commands:
            checked = "\n".join(self.cache_paths[:3])
            self.show_error("No cached G-code\nChecked:\n{}".format(checked))
            self.log("continue_unfinished: no cached commands; reset progress")
            self.save_progress(0, False)
            return

        total = len(commands)
        current_hash = self.commands_hash(commands)
        self.log(
            "continue_unfinished: cache total={} progress_count={} progress_hash={} current_hash={}".format(
                total, progress_count, progress_hash, current_hash
            )
        )

        if start_line >= total:
            self.show_error("Resume line is past\ncached G-code:\n{} / {}".format(start_line, total))
            self.log("continue_unfinished: rejected start_line >= total")
            return

        if progress_hash and progress_hash != current_hash:
            self.show_error("Cached G-code\nchanged.\nSelect file again")
            self.log("continue_unfinished: rejected hash mismatch")
            return

        if progress_count and progress_count != total:
            self.show_error("Cached G-code\nline count changed:\n{} != {}".format(progress_count, total))
            self.log("continue_unfinished: rejected count mismatch")
            return

        conf = self.menu_selector("Resume line {}?".format(start_line + 1), ["Yes", "No"])
        if conf == 0:
            mode = self.menu_selector("Start from current\nposition?", ["Yes", "No, move to XY"])
            resume_from_current = mode == 0
            self.log(
                "continue_unfinished: resume accepted, commands={} from_current={}".format(
                    len(commands), resume_from_current
                )
            )
            self.execute_gcode(
                commands,
                start_line=start_line,
                start_pen_is_down=start_pen,
                save_resume=True,
                resume_from_current=resume_from_current,
            )

    def execute_gcode(self, commands, start_line=0, start_pen_is_down=False, save_resume=True, resume_from_current=True):
        total = len(commands)
        if total == 0:
            self.show_error("No commands to print")
            return
        self.active_commands_count = total
        self.active_commands_hash = self.commands_hash(commands)
        if start_line >= total:
            self.show_error("Bad resume line\n{} / {}".format(start_line, total))
            self.log("execute_gcode: rejected start_line={} total={}".format(start_line, total))
            return
        save_every_lines = 50

        if start_line > 0:
            last_x, last_y = self.find_last_position(commands, start_line)
            if resume_from_current and hasattr(self.plotter, "set_logical_position"):
                self.plotter.set_logical_position(last_x, last_y)
                self.log(
                    "execute_gcode: resume from current physical position, start_line={} logical_x={} logical_y={}".format(
                        start_line, last_x, last_y
                    )
                )
            else:
                self.log(
                    "execute_gcode: moving to saved XY before resume, start_line={} x={} y={}".format(
                        start_line, last_x, last_y
                    )
                )
                if self.plotter.pen_is_down:
                    self.plotter.pen_up()
                self.plotter.move_to(last_x, last_y)
            if start_pen_is_down:
                self.plotter.pen_down()
            else:
                self.plotter.pen_up()
        else:
            self.plotter.pen_up()
            if start_pen_is_down:
                self.plotter.pen_down()

        for i in range(start_line, total):
            cmd = commands[i]
            if self.btn.backspace or self.btn.left or self.btn.right:
                self.plotter.pen_up()
                if save_resume:
                    self.save_progress(i, self.plotter.pen_is_down)
                self.show_error("Print Cancelled")
                while self.btn.backspace or self.btn.left or self.btn.right:
                    time.sleep(0.1)
                return

            percent = int(((i + 1) / total) * 100)
            self.lcd.clear()
            self.lcd.draw.text((5, 0), "PRINTING...", font=self.font_header)
            self.lcd.draw.text((5, 25), "Line: {}/{}".format(i + 1, total), font=self.font_item)
            self.lcd.draw.text((120, 25), "{}%".format(percent), font=self.font_item)

            bar_width = 160
            self.lcd.draw.rectangle((9, 60, 9 + bar_width, 75), outline='black')
            self.lcd.draw.rectangle((10, 61, 10 + (bar_width * percent / 100), 74), fill='black')

            self.lcd.draw.text((5, 85), cmd[:20], font=self.font_item)
            self.lcd.update()

            parts = cmd.split()
            if not parts:
                continue

            code = parts[0].upper()

            if code == "G0" or code == "G1":
                x = getattr(self.plotter, 'current_pos', [0, 0])[0]
                y = getattr(self.plotter, 'current_pos', [0, 0])[1]

                for p in parts[1:]:
                    if p.startswith('X'):
                        x = float(p[1:])
                    if p.startswith('Y'):
                        y = float(p[1:])
                self.plotter.move_to(x, y)

            elif code == "G2" or code == "G3":
                x = getattr(self.plotter, 'current_pos', [0, 0])[0]
                y = getattr(self.plotter, 'current_pos', [0, 0])[1]
                i_val = 0.0
                j_val = 0.0
                for p in parts[1:]:
                    if p.startswith('X'):
                        x = float(p[1:])
                    if p.startswith('Y'):
                        y = float(p[1:])
                    if p.startswith('I'):
                        i_val = float(p[1:])
                    if p.startswith('J'):
                        j_val = float(p[1:])
                self.plotter.draw_arc(x, y, i_val, j_val, clockwise=(code == "G2"))

            elif code == "M300":
                s_val = 0
                for p in parts[1:]:
                    if p.startswith('S'):
                        s_val = float(p[1:])
                if s_val == 30:
                    self.plotter.pen_down()
                elif s_val == 50:
                    self.plotter.pen_up()

            elif code == "G28":
                pass

            if save_resume and (((i + 1) % save_every_lines) == 0 or code == "M300"):
                self.save_progress(i + 1, self.plotter.pen_is_down)

        self.plotter.pen_up()
        if save_resume:
            self.save_progress(0, False)
        self.sound.play_tone(1000, 0.5)
        self.draw_splash("Print Done!")
        time.sleep(1)
        self.draw_splash("Calibrating X...")
        self.plotter.calibrate_x(reset_coords=True)
        time.sleep(1)

    def run_eject_paper(self):
        if not self.plotter:
            self.show_error("Plotter not ready")
            return
        self.draw_splash("Ejecting Paper...")
        self.plotter.pen_up()
        self.plotter.motor_y.on_for_degrees(SpeedPercent(-100), 3000, block=True)
        self.draw_splash("Paper Ejected")
        time.sleep(1)

    def run_calibrate_menu(self):
        while True:
            items = [
                "Full Calibrate",
                "Eject Paper",
                "Calibrate X",
                "Calibrate Paper",
                "Pen Calibrate",
                "Pen Test",
            ]
            choice = self.menu_selector("Calibration", items)
            if choice == 0:
                self.draw_splash("Calibrating...")
                self.plotter.calibrate()
            elif choice == 1:
                self.run_eject_paper()
            elif choice == 2:
                self.draw_splash("Calibrating X...")
                self.plotter.calibrate_x()
            elif choice == 3:
                self.draw_splash("Calibrating Paper...")
                self.plotter.calibrate_paper()
            elif choice == 4:
                self.run_pen_calibrate()
            elif choice == 5:
                self.run_pen_test()
            elif choice == -1:
                break

    def run_print_menu(self):
        while True:
            items = [
                "Continue Unfinished",
                "Print from Server",
                "Select File (SD/GCODE)",
            ]
            choice = self.menu_selector("Print", items)
            if choice == 0:
                self.continue_unfinished()
            elif choice == 1:
                self.run_server_mode()
            elif choice == 2:
                self.run_file_browser()
            elif choice == -1:
                break

    def main_loop(self):
        while True:
            self.cached_ip = self.get_local_ip()
            items = [
                "Print",
                "Calibrate Tools",
                "System Info",
                "Shutdown"
            ]
            choice = self.menu_selector("Plott3r Menu", items)

            if choice == 0:
                self.run_print_menu()
            elif choice == 1:
                self.run_calibrate_menu()
            elif choice == 2:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(('10.255.255.255', 1))
                    ip_addr = s.getsockname()[0]
                except Exception:
                    ip_addr = "No Network"
                s.close()
                self.show_error("IP Address:\n{}".format(ip_addr))
            elif choice == 3:
                self.draw_splash("Shutting down...")
                time.sleep(1)
                os.system("echo maker | sudo -S shutdown -h now")
            elif choice == -1:
                # Back in main menu should keep app running.
                continue


if __name__ == "__main__":
    gui = PlotterGUI()
    gui.main_loop()

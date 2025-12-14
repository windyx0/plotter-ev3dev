#!/usr/bin/env python3
import socket
import json
import time
import os
from plott3r import Plott3r
from ev3dev2.button import Button

PROGRESS_FILE = "print_progress.json"
CACHE_FILE = "gcode_cache.json"

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"progress_line": 0, "pen_is_down": False}
    try:
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return {
                "progress_line": data.get("progress_line", 0),
                "pen_is_down": data.get("pen_is_down", False)
            }
    except Exception as e:
        print("Error loading progress: {}".format(e))
        return {"progress_line": 0, "pen_is_down": False}

def save_progress(line_number, pen_is_down):
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"progress_line": line_number, "pen_is_down": pen_is_down}, f)
    except Exception as e:
        print("Error saving progress: {}".format(e))

def load_gcode_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("commands", None)
    except Exception as e:
        print("Error loading G-code cache: {}".format(e))
        return None

def save_gcode_cache(commands):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({"commands": commands}, f)
        print("Cached {} G-code commands.".format(len(commands)))
    except Exception as e:
        print("Error saving G-code cache: {}".format(e))

def find_last_position(commands, start_line):
    for i in range(start_line - 1, -1, -1):
        cmd = commands[i].strip().upper()
        if cmd.startswith(("G0", "G1")):
            parts = cmd.split()
            x = None
            y = None
            for part in parts:
                if part.startswith('X'):
                    x = float(part[1:])
                elif part.startswith('Y'):
                    y = float(part[1:])
            
            if x is not None and y is not None:
                return (x, y)
            
    return (0, 0)

def run_print_job(plotter, commands, start_line, start_pen_is_down):
    print("--- Starting print job from line {} ---".format(start_line + 1))
    total_lines = len(commands)

    if start_line > 0:
        print("Resuming print. Finding last known position...")
        last_x, last_y = find_last_position(commands, start_line)
        print("Last position was ({}, {}). Moving there...".format(last_x, last_y))
        
        if plotter.pen_is_down:
            plotter.pen_up()
            
        plotter.move_to(last_x, last_y)
        
        print("Setting initial pen state to: {}".format("DOWN" if start_pen_is_down else "UP"))
        if start_pen_is_down:
            plotter.pen_down()
        
    else:
        plotter.pen_up()
    
    line_number = start_line
    try:
        for i in range(start_line, total_lines):
            line_number = i
            cmd = commands[line_number].strip().upper()
            if not cmd:
                continue

            print("Executing [{} / {}]: {}".format(line_number + 1, total_lines, cmd))

            if cmd == "M300 S30":
                plotter.pen_down()
            elif cmd == "M300 S50":
                plotter.pen_up()
            elif cmd.startswith(("G0", "G1")):
                parts = cmd.split()
                x = plotter.current_pos[0]
                y = plotter.current_pos[1]
                for part in parts:
                    if part.startswith('X'):
                        x = float(part[1:])
                    elif part.startswith('Y'):
                        y = float(part[1:])
                plotter.move_to(x, y)
            elif cmd == "G28":
                plotter.pen_up()
                plotter.move_to(0, 0)
            elif cmd.startswith(("G21", "G90", "G92")):
                print("Ignoring: {}".format(cmd))

            save_progress(line_number + 1, plotter.pen_is_down)

        print("--- PRINT JOB SUCCESS ---")
        save_progress(0, False)

    except Exception as e:
        print("\n--- PRINT JOB FAILED ---")
        if line_number < len(commands):
            print("Error during command {}: {}".format(commands[line_number], e))
        else:
            print("Error after last command: {}".format(e))
    
    print("Print job finished.")

def handle_connection(conn, addr, plotter):
    print("Client connected:", addr)
    commands = []
    try:
        data = b""
        while True:
            part = conn.recv(4096)
            if not part:
                break
            data += part
        
        data_str = data.decode('utf-8').strip()
        received = json.loads(data_str)
        
        if not isinstance(received, dict) or 'commands' not in received:
            raise ValueError("Invalid data format")
        
        commands = received['commands']
        print("Received {} commands. Caching...".format(len(commands)))

        save_gcode_cache(commands)
        save_progress(0, False)
        
        conn.sendall(b"OK")
        print("Sent OK to client.")
        
    except Exception as e:
        print("Error in connection handler: {}".format(e))
    finally:
        return commands 


def run_server(plotter):
    host = '0.0.0.0'
    port = 15614
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        print("Server running on {}:{}, waiting for new jobs...".format(host, port))
        
        while True:
            conn, addr = None, None
            commands_to_run = []
            try:
                print("\nWaiting for new connection...")
                conn, addr = s.accept()
                commands_to_run = handle_connection(conn, addr, plotter)
                
            except Exception as e:
                print("Server error during connection: {}".format(str(e)))
            finally:
                if conn:
                    conn.close() 
                    print("Connection closed.")
            
            if commands_to_run:
                print("Starting new print job...")
                run_print_job(plotter, commands_to_run, 0, False)
                plotter.calibrate()
                print("Job finished. (Ready for next job)")
            else:
                print("No commands received. (Ready for next job)")

if __name__ == "__main__":
    plotter = Plott3r()
    plotter.calibrate()
    btn = Button()
    
    progress_data = load_progress()
    progress_line = progress_data.get("progress_line", 0)
    pen_is_down = progress_data.get("pen_is_down", False)
    
    if progress_line > 0:
        print("--- RESUME ---")
        print("Previous job found, stopped at line {}".format(progress_line))
        print("Last pen state was: {}".format("DOWN" if pen_is_down else "UP"))
        
        print("Continue? (LEFT = Yes, RIGHT = No)")
        choice = ""
        while choice not in ['Y', 'N']:
            
            if btn.left:
                choice = 'Y'
                print("LEFT pressed (Y). Resuming...")
                while btn.left:
                    time.sleep(0.01)

            if btn.right:
                choice = 'N'
                print("RIGHT pressed (N). Discarding...")
                while btn.right:
                    time.sleep(0.01)

            time.sleep(0.01)
        
        if choice == 'Y':
            print("Loading cached G-code...")
            cached_commands = load_gcode_cache()
            if cached_commands:
                run_print_job(plotter, cached_commands, progress_line, pen_is_down)
            else:
                print("Cache file not found. Starting new server.")
                save_progress(0, False)
        else:
             save_progress(0, False)

    run_server(plotter)
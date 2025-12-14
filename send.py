import socket
import json
import sys
import struct
from time import sleep
import argparse


def send_gcode_file(filename, host, port=15614):
    try:
        with open(filename, 'r') as f:
            commands = [line.strip() for line in f if line.strip() and not line.startswith(';')]

        if not commands:
            print("Error: G-code file is empty")
            return False

        data = {
            'version': 1,
            'commands': commands
        }
        json_data = json.dumps(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(15)

            for attempt in range(3):
                try:
                    print("Connecting to {}:{} (attempt {})...".format(host, port, attempt + 1))
                    s.connect((host, port))
                    break
                except Exception as e:
                    if attempt == 2:
                        print("Connection failed:", str(e))
                        return False
                    sleep(2)

            print("Sending {} commands...".format(len(commands)))
            s.sendall(json_data.encode('utf-8'))

            response = s.recv(1024).decode()
            print("Server response:", response)
            return response == "OK"

    except Exception as e:
        print("Error:", str(e))
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send G-code file to EV3 Plott3r")
    parser.add_argument("gcode_file", help="Path to the G-code file")
    parser.add_argument("--host", required=True, help="IP address or hostname (e.g., 'ev3dev' or '192.168.1.100')")
    parser.add_argument("--port", type=int, default=15614, help="Port number (default: 15614)")

    args = parser.parse_args()

    print("Sending G-code from {} to {}:{}".format(args.gcode_file, args.host, args.port))
    if not send_gcode_file(args.gcode_file, args.host, args.port):
        print("Failed to send G-code.")
        sys.exit(1)

    print("G-code sent successfully.")
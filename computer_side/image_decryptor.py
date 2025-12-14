import matplotlib.pyplot as plt
import re


class GCodeVisualizer:
    def __init__(self):
        self.segments = []
        self.current_x = 0
        self.current_y = 0
        self.pen_down = False
        self.current_color = 'black'

    def process_line(self, line):
        line = line.strip()
        if not line or line.startswith(';'):
            return

        if line.startswith('M300'):
            if 'S30' in line:
                self.pen_down = True
                self.current_color = 'black'
            elif 'S50' in line:
                self.pen_down = False
                self.current_color = 'red'
            return

        if line.startswith(('G0', 'G1')):
            x_match = re.search(r'X([\d\.]+)', line)
            y_match = re.search(r'Y([\d\.]+)', line)

            new_x = float(x_match.group(1)) if x_match else self.current_x
            new_y = float(y_match.group(1)) if y_match else self.current_y

            self.segments.append({
                'x': [self.current_x, new_x],
                'y': [self.current_y, new_y],
                'color': self.current_color
            })

            self.current_x = new_x
            self.current_y = new_y

    def plot(self):
        plt.figure(figsize=(12, 12))

        for segment in self.segments:
            plt.plot(segment['x'], segment['y'],
                     color=segment['color'],
                     linewidth=1,
                     marker='o' if segment['color'] == 'red' else '',
                     markersize=3)

        plt.gca().invert_yaxis()
        plt.title('G-code Visualization')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.grid(True)
        plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize G-code file')
    parser.add_argument('filename', help='Path to G-code file')
    args = parser.parse_args()

    visualizer = GCodeVisualizer()

    try:
        with open(args.filename, 'r') as f:
            for line in f:
                visualizer.process_line(line)
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        return

    visualizer.plot()


if __name__ == "__main__":
    main()
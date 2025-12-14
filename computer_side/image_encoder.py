#!/usr/bin/env python3
import cv2
import numpy as np
from skimage.filters import threshold_otsu
import argparse
import json
from typing import List, Any, Dict
import io
import math

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    from PIL import Image

    SVG_SUPPORT = True
except ImportError:
    print("Warning: 'svglib' or 'reportlab' not found. SVG support is disabled.")
    print("Install with: pip install svglib")
    SVG_SUPPORT = False


def load_image_grayscale(image_path: str) -> np.ndarray:
    """Loads any image (PNG, JPG, SVG) as a grayscale numpy array"""
    if image_path.lower().endswith(".svg"):
        if not SVG_SUPPORT:
            raise ImportError("SVG file detected, but 'svglib' library is missing.")
        try:
            drawing = svg2rlg(image_path)
            if drawing is None: raise ValueError("Could not parse SVG.")
            png_data = renderPM.drawToString(drawing, fmt='PNG')
            img_pil = Image.open(io.BytesIO(png_data)).convert('L')
            return np.array(img_pil)
        except Exception as e:
            raise ValueError(f"Error rendering SVG {image_path}: {str(e)}")
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {image_path} (is it PNG or JPG?)")
        return img


def get_distance(p1, p2):
    """Calculates the Euclidean distance between two points (x, y)"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_line_gcode(contours: List[np.ndarray], config: dict) -> List[str]:
    """
    (VECTOR / EDGE MODE) Conversion of contours/lines to G-code WITH PATH OPTIMIZATION (NEAREST NEIGHBOR)
    """
    gcode = ["G21", "G90", "G28"]
    feedrate = config.get('feedrate', 1000)
    rapid_feedrate = config.get('rapid_feedrate', 2000)

    unvisited_contours = []
    for contour in contours:
        start_point = (contour[0][0][0], contour[0][0][1])
        unvisited_contours.append((start_point, contour))

    sorted_contours = []
    current_pos = (0, 0)

    print(f"Optimizing path for {len(unvisited_contours)} lines (Nearest Neighbor)...")

    while unvisited_contours:
        closest_index = -1
        min_dist = float('inf')

        for i, (start_point, _) in enumerate(unvisited_contours):
            dist = get_distance(current_pos, start_point)
            if dist < min_dist:
                min_dist = dist
                closest_index = i

        start_point, closest_contour = unvisited_contours.pop(closest_index)
        sorted_contours.append(closest_contour)

        end_point_np = closest_contour[-1][0]
        current_pos = (end_point_np[0], end_point_np[1])

    print("Path optimization complete.")

    for contour in sorted_contours:
        x, y = contour[0][0]
        gcode.append(f"G0 X{x:.2f} Y{y:.2f} F{rapid_feedrate}")
        gcode.append(config.get('pen_down', 'M300 S30'))

        for point in contour[1:]:
            x, y = point[0]
            gcode.append(f"G1 X{x:.2f} Y{y:.2f} F{feedrate}")

        gcode.append(config.get('pen_up', 'M300 S50'))

    gcode.append("G28")
    return gcode


def process_image_to_vectors(img: np.ndarray) -> Dict[str, Any]:
    """(VECTOR MODE) Finds the OUTER edges of solid shapes."""
    thresh = threshold_otsu(img)
    binary = (img < thresh).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return {'contours': contours}


def apply_floyd_steinberg_dither(img_float: np.ndarray) -> np.ndarray:
    height, width = img_float.shape
    img_out = np.copy(img_float)
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = img_out[y, x]
            new_pixel = 1.0 if old_pixel > 0.5 else 0.0
            img_out[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            img_out[y, x + 1] += quant_error * 7 / 16
            img_out[y + 1, x - 1] += quant_error * 3 / 16
            img_out[y + 1, x] += quant_error * 5 / 16
            img_out[y + 1, x + 1] += quant_error * 1 / 16
    return img_out


def process_image_for_hatching(img: np.ndarray) -> np.ndarray:
    """(HATCH MODE) Preparing an image for hatching (dithering)"""
    img_inverted = 255 - img
    img_float = img_inverted.astype(float) / 255.0
    dithered_img = apply_floyd_steinberg_dither(img_float)
    return dithered_img


def generate_hatching_gcode(dithered_img: np.ndarray, config: dict) -> List[str]:
    """(HATCH MODE) Converting dithered image to G-code"""
    gcode = ["G21", "G90", "G28"]
    feedrate = config.get('feedrate', 1000)
    rapid_feedrate = config.get('rapid_feedrate', 2000)
    line_spacing_px = config.get('line_spacing_px', 3)
    mm_per_px_x = config.get('mm_per_px_x', 0.1)
    mm_per_px_y = config.get('mm_per_px_y', 0.1)
    height, width = dithered_img.shape
    gcode.append(config.get('pen_up', 'M300 S50'))
    pen_is_currently_down = False

    for y_px in range(0, height, line_spacing_px):
        y_mm = y_px * mm_per_px_y
        if (y_px // line_spacing_px) % 2 == 0:
            x_range = range(width)
        else:
            x_range = range(width - 1, -1, -1)

        if pen_is_currently_down:
            gcode.append(config.get('pen_up', 'M300 S50'))
            pen_is_currently_down = False

        for x_px in x_range:
            pixel_brightness = dithered_img[y_px, x_px]
            x_mm = x_px * mm_per_px_x
            if pixel_brightness < 0.5:
                if not pen_is_currently_down:
                    gcode.append(f"G0 X{x_mm:.2f} Y{y_mm:.2f} F{rapid_feedrate}")
                    gcode.append(config.get('pen_down', 'M300 S30'))
                    pen_is_currently_down = True
                else:
                    gcode.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feedrate}")
            else:
                if pen_is_currently_down:
                    gcode.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feedrate}")
                    gcode.append(config.get('pen_up', 'M300 S50'))
                    pen_is_currently_down = False

    if pen_is_currently_down:
        gcode.append(config.get('pen_up', 'M300 S50'))
    gcode.append("G28")
    return gcode


def process_image_to_edges(img: np.ndarray, config: dict) -> Dict[str, Any]:
    """(NEW 'edge' MODE) Finds LINES with Canny Edge Detector."""

    low_threshold = config.get('canny_low', 50)
    high_threshold = config.get('canny_high', 150)
    edges = cv2.Canny(img, low_threshold, high_threshold)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    simplified_contours = []
    min_len = config.get('min_line_length', 5)

    for contour in contours:
        if len(contour) < min_len:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 2:
            simplified_contours.append(approx)

    return {'contours': simplified_contours}


def main():
    parser = argparse.ArgumentParser(description='Image to G-code Converter')
    parser.add_argument("image_path", help="Path to input image (JPG, PNG, or SVG)")
    parser.add_argument("--output", default="output.gcode", help="Output G-code file")
    parser.add_argument("--config", default="config.json", help="Config file (optional)")

    parser.add_argument("--mode", default=None, choices=["hatch", "vector", "edge"],
                        help="Processing mode: 'hatch' (штриховка), 'vector' (контуры), 'edge' (линии/эскиз). Default: 'vector' for SVG, 'edge' for others")

    parser.add_argument("--width_mm", type=float, default=150.0,
                        help="[Hatch/Edge/Vector] Physical width of the final drawing in mm")

    parser.add_argument("--spacing_px", type=int, default=3,
                        help="[Hatch] Pixel spacing between horizontal lines")

    parser.add_argument("--resolution_x", type=int, default=800,
                        help="[Hatch/Edge/Vector] Horizontal processing resolution in pixels")

    parser.add_argument("--canny_low", type=int, default=50,
                        help="[Edge] Canny low threshold (lower = more lines)")
    parser.add_argument("--canny_high", type=int, default=150,
                        help="[Edge] Canny high threshold (higher = less 'weak' lines)")
    parser.add_argument("--min_line_length", type=int, default=5,
                        help="[Edge] Minimum line length in pixels to draw (removes noise)")

    args = parser.parse_args()

    try:
        config = {
            "feedrate": 1000,
            "rapid_feedrate": 2000,
            "pen_down": "M300 S30",
            "pen_up": "M300 S50"
        }
        config.update(vars(args))

        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                config.update(user_config)
        except FileNotFoundError:
            print(f"Note: Config file {args.config} not found, using defaults")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in config file {args.config}, using defaults")

        run_mode = args.mode
        if run_mode is None:
            if args.image_path.lower().endswith(".svg"):
                run_mode = "vector"
                print("SVG file detected. Auto-selecting 'vector' (контуры) mode.")
            else:
                run_mode = "edge"
                print("Raster image detected. Auto-selecting 'edge' (линии/эскиз) mode.")
        else:
            print(f"User selected '{run_mode}' mode.")

        original_img_gray = load_image_grayscale(args.image_path)
        original_height, original_width = original_img_gray.shape
        aspect_ratio = original_height / original_width

        resolution_y = int(args.resolution_x * aspect_ratio)
        output_size_px = (args.resolution_x, resolution_y)

        height_mm = args.width_mm * aspect_ratio
        mm_per_px_x = args.width_mm / args.resolution_x
        mm_per_px_y = height_mm / resolution_y

        print(f"Original size: {original_width}x{original_height} px")
        print(f"Processing size: {output_size_px[0]}x{output_size_px[1]} px")
        print(f"Physical size: {args.width_mm:.1f}x{height_mm:.1f} mm")

        resized_img = cv2.resize(original_img_gray, output_size_px)

        config['mm_per_px_x'] = mm_per_px_x
        config['mm_per_px_y'] = mm_per_px_y

        if run_mode == 'hatch':
            print(f"Line spacing: {args.spacing_px} px ({args.spacing_px * mm_per_px_y:.2f} mm)")
            config['line_spacing_px'] = args.spacing_px
            dithered_img = process_image_for_hatching(resized_img)
            gcode = generate_hatching_gcode(dithered_img, config)

        elif run_mode == 'vector':
            print("Processing 'vector' (outlines)...")
            result = process_image_to_vectors(resized_img)
            contours = result['contours']
            print(f"Found {len(contours)} contours.")
            gcode = generate_line_gcode(contours, config)

        else:
            print("Processing 'edge' (line-art)...")
            print(f"Canny thresholds: Low={args.canny_low}, High={args.canny_high}")
            result = process_image_to_edges(resized_img, config)
            contours = result['contours']
            print(f"Found {len(contours)} lines.")
            gcode = generate_line_gcode(contours, config)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("\n".join(gcode))
        print(f"Success: G-code saved to {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
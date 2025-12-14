#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import argparse
import json
import io
import math
import base64
import socket
import sys
import time
import threading
from typing import List, Any, Dict

try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print("Ошибка: 'Flask' не найден. Пожалуйста, установите его: pip install Flask")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    from skimage.filters import threshold_otsu
except ImportError:
    print(
        "Ошибка: 'opencv-python' или 'scikit-image' не найдены. Установите их: pip install opencv-python scikit-image")
    sys.exit(1)

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    from PIL import Image


    SVG_SUPPORT = True
except ImportError:
    print("Warning: 'svglib' or 'reportlab' не найдены. SVG поддержка отключена.")
    SVG_SUPPORT = False

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
except ImportError:
    print("Ошибка: 'matplotlib' не найден. Установите его: pip install matplotlib")
    sys.exit(1)


def load_image_grayscale(image_path: str, file_storage=None) -> np.ndarray:
    if file_storage:
        in_memory_file = io.BytesIO(file_storage.read())
    else:
        in_memory_file = image_path

    if image_path.lower().endswith(".svg"):
        if not SVG_SUPPORT:
            raise ImportError("SVG file detected, but 'svglib' library is missing.")
        try:
            drawing = svg2rlg(in_memory_file)
            if drawing is None: raise ValueError("Could not parse SVG.")
            png_data = renderPM.drawToString(drawing, fmt='PNG')
            img_pil = Image.open(io.BytesIO(png_data)).convert('L')
            return np.array(img_pil)
        except Exception as e:
            raise ValueError(f"Error rendering SVG {image_path}: {str(e)}")
    else:
        if file_storage:
            img_pil = Image.open(in_memory_file).convert('L')
            img = np.array(img_pil)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        return img


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_line_gcode(contours: List[np.ndarray], config: dict) -> List[str]:
    gcode = ["G21", "G90", "G28"]
    feedrate = config.get('feedrate', 1000)
    rapid_feedrate = config.get('rapid_feedrate', 2000)

    unvisited_contours = []
    for contour in contours:
        start_point = (contour[0][0][0], contour[0][0][1])
        end_point = (contour[-1][0][0], contour[-1][0][1])
        unvisited_contours.append([start_point, end_point, contour, False])

    sorted_contours = []
    current_pos = (0, 0)

    while unvisited_contours:
        closest_index = -1
        min_dist = float('inf')
        should_reverse = False

        for i, (start_point, end_point, _, _) in enumerate(unvisited_contours):
            dist_to_start = get_distance(current_pos, start_point)
            dist_to_end = get_distance(current_pos, end_point)

            if dist_to_start < min_dist:
                min_dist = dist_to_start
                closest_index = i
                should_reverse = False

            if dist_to_end < min_dist:
                min_dist = dist_to_end
                closest_index = i
                should_reverse = True

        start_point, end_point, closest_contour, _ = unvisited_contours.pop(closest_index)

        if should_reverse:
            sorted_contours.append(np.flip(closest_contour, axis=0))
            current_pos = start_point
        else:
            sorted_contours.append(closest_contour)
            current_pos = end_point

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
    img_inverted = 255 - img
    img_float = img_inverted.astype(float) / 255.0
    dithered_img = apply_floyd_steinberg_dither(img_float)
    return dithered_img


def generate_hatching_gcode(dithered_img: np.ndarray, config: dict) -> List[str]:
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
    if pen_is_currently_down: gcode.append(config.get('pen_up', 'M300 S50'))
    gcode.append("G28")
    return gcode


def process_image_to_edges(img: np.ndarray, config: dict) -> Dict[str, Any]:
    low_threshold = config.get('canny_low', 50)
    high_threshold = config.get('canny_high', 150)
    edges = cv2.Canny(img, low_threshold, high_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []
    min_len = config.get('min_line_length', 5)
    for contour in contours:
        if len(contour) < min_len: continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 2:
            simplified_contours.append(approx)
    return {'contours': simplified_contours}


def create_gcode_visualization(gcode_lines: List[str]) -> str:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    current_x, current_y = 0, 0
    pen_down = False
    for line in gcode_lines:
        line = line.strip().upper()
        if not line or line.startswith(';'): continue
        if line == "M300 S30": pen_down = True; continue
        if line == "M300 S50": pen_down = False; continue
        if line.startswith(("G0", "G1")):
            try:
                parts = line.split()
                new_x, new_y = current_x, current_y
                for part in parts:
                    if part.startswith('X'):
                        new_x = float(part[1:])
                    elif part.startswith('Y'):
                        new_y = float(part[1:])
                color = 'black' if pen_down else 'red'
                linewidth = 1.0 if pen_down else 0.5
                ax.plot([current_x, new_x], [current_y, new_y], color=color, linewidth=linewidth)
                current_x, current_y = new_x, new_y
            except Exception:
                pass
    ax.set_title("G-code Visualization")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return "data:image/png;base64," + img_base64


def send_gcode_to_ev3(commands: List[str], host='ev3dev', port=15614):
    if not commands:
        raise ValueError("G-code пуст, нечего отправлять.")
    data = {'version': 1, 'commands': commands}
    json_data = json.dumps(data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(60.0)
        print(f"Connecting to {host}:{port}...")
        s.connect((host, port))
        print("Sending G-code data...")
        s.sendall(json_data.encode('utf-8'))
        response = s.recv(1024).decode('utf-8')
        print(f"EV3 Response: {response}")
        if response != "OK":
            raise Exception(f"EV3 вернул ошибку: {response}")
    return "OK"


app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plott3r PC Control</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        input[type="radio"]:checked + label {
            background-color: #3b82f6; /* bg-blue-600 */
            color: white;
            border-color: #3b82f6; /* border-blue-600 */
        }
        input[type="file"]::file-selector-button {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #2563eb;
        }
        #gcode-preview {
            font-family: monospace;
            font-size: 12px;
            white-space: pre;
            overflow-y: scroll;
            height: 400px;
            background-color: #1f2937; /* bg-gray-800 */
            border: 1px solid #374151; /* border-gray-700 */
            border-radius: 0.375rem;
        }
    </style>
</head>
<body class="bg-gray-900 text-white p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-center mb-6">Plott3r PC Control</h1>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">

            <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
                <form id="upload-form">
                    <div class="mb-6">
                        <label class="block text-lg font-medium mb-2" for="image_file">1. Загрузите изображение</label>
                        <input class="block w-full text-sm text-gray-400 border border-gray-700 rounded-lg cursor-pointer bg-gray-700 focus:outline-none" 
                               id="image_file" name="image_file" type="file" accept="image/png, image/jpeg, image/svg+xml" required>
                        <p class="mt-1 text-xs text-gray-400">Поддерживаются PNG, JPG, SVG.</p>
                    </div>

                    <div class="mb-6">
                        <label class="block text-lg font-medium mb-2">2. Выберите режим</label>
                        <div class="grid grid-cols-3 gap-2">
                            <input type="radio" name="mode" id="mode-edge" value="edge" class="hidden" checked onchange="toggleSettings()">
                            <label for="mode-edge" class="text-center p-3 border border-gray-700 rounded-lg cursor-pointer transition">Линии (Эскиз)</label>

                            <input type="radio" name="mode" id="mode-vector" value="vector" class="hidden" onchange="toggleSettings()">
                            <label for="mode-vector" class="text-center p-3 border border-gray-700 rounded-lg cursor-pointer transition">Контуры (Лого)</label>

                            <input type="radio" name="mode" id="mode-hatch" value="hatch" class="hidden" onchange="toggleSettings()">
                            <label for="mode-hatch" class="text-center p-3 border border-gray-700 rounded-lg cursor-pointer transition">Штриховка (Фото)</label>
                        </div>
                    </div>

                    <div class="mb-6">
                        <label class="block text-lg font-medium mb-2">3. Настройте параметры</label>

                        <div id="settings-edge" class="space-y-4">
                            <div>
                                <label for="canny_low" class="block text-sm">Canny Low (Больше линий: <span id="canny_low_val">50</span>)</label>
                                <input type="range" id="canny_low" name="canny_low" min="10" max="200" value="50" class="w-full" oninput="updateSlider('canny_low_val', this.value)">
                            </div>
                            <div>
                                <label for="canny_high" class="block text-sm">Canny High (Меньше линий: <span id="canny_high_val">150</span>)</label>
                                <input type="range" id="canny_high" name="canny_high" min="50" max="400" value="150" class="w-full" oninput="updateSlider('canny_high_val', this.value)">
                            </div>
                        </div>

                        <div id="settings-hatch" class="space-y-4" style="display: none;">
                            <div>
                                <label for="spacing_px" class="block text-sm">Расстояние (Плотность: <span id="spacing_px_val">3</span> px)</label>
                                <input type="range" id="spacing_px" name="spacing_px" min="1" max="10" value="3" class="w-full" oninput="updateSlider('spacing_px_val', this.value)">
                            </div>
                        </div>

                        <div id="settings-vector" class="text-gray-400" style="display: none;">
                            <p>Дополнительные настройки не требуются.</p>
                        </div>

                        <div class="mt-4">
                            <label for="width_mm" class="block text-sm">Ширина рисунка (мм): <span id="width_mm_val">150</span></label>
                            <input type="range" id="width_mm" name="width_mm" min="50" max="500" value="150" class="w-full" oninput="updateSlider('width_mm_val', this.value)">
                        </div>
                    </div>

                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition text-lg">
                        Обработать и Показать G-code
                    </button>
                </form>
            </div>

            <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
                <div class="flex border-b border-gray-700 mb-4">
                    <button id="tab-btn-preview" class="py-2 px-4 font-medium text-blue-400 border-b-2 border-blue-400">Предпросмотр</button>
                    <button id="tab-btn-gcode" class="py-2 px-4 font-medium text-gray-400">G-code (<span id="gcode-lines">0</span> строк)</button>
                </div>

                <div id="tab-content-preview" class="tab-content active">
                    <div id="preview-placeholder" class="h-[400px] flex items-center justify-center bg-gray-700 rounded-lg text-gray-400">
                        Загрузите изображение и нажмите "Обработать"
                    </div>
                    <img id="preview-image" src="" alt="G-code Preview" class="w-full h-auto rounded-lg" style="display: none;">
                </div>

                <div id="tab-content-gcode" class="tab-content">
                    <pre id="gcode-preview" class="text-gray-300">G-code будет здесь...</pre>
                </div>
            </div>
        </div>

        <div class="bg-gray-800 p-6 rounded-lg shadow-lg mt-6">
            <h2 class="text-xl font-bold mb-4">5. Отправка на EV3</h2>
            <div class="flex flex-col md:flex-row gap-4">
                <input type="text" id="ev3-ip" placeholder="IP-адрес EV3 (напр. 192.168.1.100)" class="flex-grow bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500">
                <button id="send-button" class="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition" disabled>
                    Отправить на EV3
                </button>
            </div>
        </div>

        <div id="status-overlay" style="display: none;" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50">
            <div class="flex items-center space-x-3 bg-gray-800 p-6 rounded-lg shadow-2xl">
                <svg class="animate-spin h-8 w-8 text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span id="status-text" class="text-xl font-medium">Обработка...</span>
            </div>
        </div>
    </div>

    <script>
        let GCODE_CACHE = []; 

        function updateSlider(id, value) {
            document.getElementById(id).innerText = value;
        }

        function toggleSettings() {
            const mode = document.querySelector('input[name="mode"]:checked').value;
            document.getElementById('settings-edge').style.display = (mode === 'edge') ? 'block' : 'none';
            document.getElementById('settings-vector').style.display = (mode === 'vector') ? 'block' : 'none';
            document.getElementById('settings-hatch').style.display = (mode === 'hatch') ? 'block' : 'none';
        }

        const tabBtnPreview = document.getElementById('tab-btn-preview');
        const tabBtnGcode = document.getElementById('tab-btn-gcode');
        const tabContentPreview = document.getElementById('tab-content-preview');
        const tabContentGcode = document.getElementById('tab-content-gcode');

        tabBtnPreview.onclick = () => {
            tabBtnPreview.classList.add('text-blue-400', 'border-blue-400');
            tabBtnGcode.classList.remove('text-blue-400', 'border-blue-400');
            tabContentPreview.classList.add('active');
            tabContentGcode.classList.remove('active');
        };
        tabBtnGcode.onclick = () => {
            tabBtnGcode.classList.add('text-blue-400', 'border-blue-400');
            tabBtnPreview.classList.remove('text-blue-400', 'border-blue-400');
            tabContentGcode.classList.add('active');
            tabContentPreview.classList.remove('active');
        };

        const statusOverlay = document.getElementById('status-overlay');
        const statusText = document.getElementById('status-text');
        function showStatus(text) {
            statusText.innerText = text;
            statusOverlay.style.display = 'flex';
        }
        function hideStatus() {
            statusOverlay.style.display = 'none';
        }

        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const fileInput = document.getElementById('image_file');
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Пожалуйста, выберите файл изображения.');
                return;
            }
            showStatus('Обработка изображения...');
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Ошибка сервера');
                }
                const result = await response.json();
                document.getElementById('preview-placeholder').style.display = 'none';
                document.getElementById('preview-image').src = result.preview_image;
                document.getElementById('preview-image').style.display = 'block';
                GCODE_CACHE = result.gcode_lines;
                document.getElementById('gcode-preview').innerText = GCODE_CACHE.join('\\n');
                document.getElementById('gcode-lines').innerText = GCODE_CACHE.length;
                document.getElementById('send-button').disabled = false;
                document.getElementById('send-button').classList.remove('bg-green-600', 'hover:bg-green-700');
                document.getElementById('send-button').classList.add('bg-blue-600', 'hover:bg-blue-700');
                tabBtnPreview.click();
            } catch (error) {
                alert('Ошибка обработки: ' + error.message);
            } finally {
                hideStatus();
            }
        });

        document.getElementById('send-button').addEventListener('click', async () => {
            const ip = document.getElementById('ev3-ip').value;
            if (!ip) {
                alert('Пожалуйста, введите IP-адрес EV3.');
                return;
            }
            if (GCODE_CACHE.length === 0) {
                alert('Нет G-code для отправки. Сначала обработайте изображение.');
                return;
            }
            showStatus('Отправка на EV3...');
            try {
                const response = await fetch('/send_to_ev3', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        commands: GCODE_CACHE,
                        host: ip,
                        port: 15614
                    })
                });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Ошибка отправки');
                }
                const result = await response.json();
                showStatus('Успешно отправлено!');
                document.getElementById('send-button').classList.add('bg-green-600', 'hover:bg-green-700');
                document.getElementById('send-button').classList.remove('bg-blue-600', 'hover:bg-blue-700');
                setTimeout(hideStatus, 2000);
            } catch (error) {
                alert('Ошибка отправки: ' + error.message);
                hideStatus();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process_image_route():
    try:
        if 'image_file' not in request.files:
            return jsonify({"error": "Файл изображения не найден"}), 400
        file = request.files['image_file']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        config = {
            "feedrate": 1000, "rapid_feedrate": 2000,
            "pen_down": "M300 S30", "pen_up": "M300 S50",
            "canny_low": int(request.form.get('canny_low', 50)),
            "canny_high": int(request.form.get('canny_high', 150)),
            "line_spacing_px": int(request.form.get('spacing_px', 3)),
            "min_line_length": 5,
            "width_mm": float(request.form.get('width_mm', 150.0)),
            "resolution_x": 800
        }
        run_mode = request.form.get('mode')
        if run_mode is None:
            if file.filename.lower().endswith(".svg"):
                run_mode = "vector"
            else:
                run_mode = "edge"

        original_img_gray = load_image_grayscale(file.filename, file_storage=file)
        original_height, original_width = original_img_gray.shape
        aspect_ratio = original_height / original_width
        resolution_y = int(config['resolution_x'] * aspect_ratio)
        output_size_px = (config['resolution_x'], resolution_y)
        height_mm = config['width_mm'] * aspect_ratio
        config['mm_per_px_x'] = config['width_mm'] / config['resolution_x']
        config['mm_per_px_y'] = height_mm / resolution_y
        resized_img = cv2.resize(original_img_gray, output_size_px)

        gcode_lines = []
        if run_mode == 'hatch':
            dithered_img = process_image_for_hatching(resized_img)
            gcode_lines = generate_hatching_gcode(dithered_img, config)
        elif run_mode == 'vector':
            result = process_image_to_vectors(resized_img)
            gcode_lines = generate_line_gcode(result['contours'], config)
        else:
            result = process_image_to_edges(resized_img, config)
            gcode_lines = generate_line_gcode(result['contours'], config)

        if not gcode_lines:
            return jsonify({"error": "Не удалось сгенерировать G-code. Изображение может быть пустым."}), 400

        preview_image_base64 = create_gcode_visualization(gcode_lines)

        return jsonify({
            "message": "Успешно обработано",
            "gcode_lines": gcode_lines,
            "line_count": len(gcode_lines),
            "preview_image": preview_image_base64
        })

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
        return jsonify({"error": str(e)}), 500


def send_gcode_task(commands: List[str], host: str, port: int):
    try:
        print(f"[Thread] Запуск отправки на {host}:{port}...")
        send_gcode_to_ev3(commands, host, port)
        print(f"[Thread] Отправка на {host}:{port} успешно завершена.")
    except Exception as e:
        if str(e) == "timed out":
            print(f"[Thread] Отправка на {host}:{port} успешно завершена.")
        else:
            print(f"[Thread] ОШИБКА отправки на {host}:{port}: {str(e)}")


@app.route('/send_to_ev3', methods=['POST'])
def send_to_ev3_route():
    try:
        data = request.json
        commands = data.get('commands')
        host = data.get('host')
        port = data.get('port', 15614)
        if not commands or not host:
            return jsonify({"error": "Отсутствует G-code или IP-адрес"}), 400

        thread = threading.Thread(
            target=send_gcode_task,
            args=(commands, host, port),
            daemon=True
        )
        thread.start()

        print(f"Отвечаем браузеру 'OK', запускаем отправку на {host} в фоне...")
        return jsonify({"message": "Отправка запущена!"})

    except Exception as e:
        print(f"Ошибка при запуске потока отправки: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("--- Plott3r PC App Server ---")
    print("Запуск... Откройте http://127.0.0.1:5000 в вашем браузере.")
    print("Нажмите Ctrl+C, чтобы остановить сервер.")
    app.run(host='0.0.0.0', port=5000)
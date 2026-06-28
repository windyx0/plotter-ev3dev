#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import re
import io
import json
import math
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from skimage.filters import threshold_otsu

try:
    import svgelements
    SVG_SUPPORT = True
except Exception:
    SVG_SUPPORT = False

matplotlib.use("Agg")


app = Flask(__name__, template_folder=".")


def has_cuda_acceleration() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def resize_image(img: np.ndarray, output_size_px: Tuple[int, int], use_cuda: bool) -> np.ndarray:
    if use_cuda:
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img)
            resized_gpu = cv2.cuda.resize(gpu, output_size_px, interpolation=cv2.INTER_AREA)
            return resized_gpu.download()
        except Exception:
            pass
    return cv2.resize(img, output_size_px, interpolation=cv2.INTER_AREA)


def preprocess_for_edges(img: np.ndarray) -> np.ndarray:
    # Slight denoise + local contrast enhancement keeps meaningful edges on photos.
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced


def estimate_canny_thresholds(img_gray: np.ndarray) -> Tuple[int, int]:
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    non_zero = grad[grad > 1e-3]
    if non_zero.size == 0:
        return 35, 110

    p40 = float(np.percentile(non_zero, 40))
    p82 = float(np.percentile(non_zero, 82))
    low = int(np.clip(p40 * 0.85, 18, 180))
    high = int(np.clip(p82 * 1.10, 60, 320))
    if high <= low + 20:
        high = min(320, low + 30)
    return low, high


def load_image_grayscale(filename: str, file_storage=None) -> np.ndarray:
    if file_storage:
        in_memory_file = io.BytesIO(file_storage.read())
    else:
        in_memory_file = filename

    if file_storage:
        img_pil = Image.open(in_memory_file).convert("L")
        img = np.array(img_pil)
    else:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image from {filename}")
    return img


def parse_svg_to_contours(file_storage) -> Tuple[List[np.ndarray], float, float]:
    if not SVG_SUPPORT:
        raise ImportError("svgelements is not installed.")
    
    in_memory_file = io.BytesIO(file_storage.read())
    svg = svgelements.SVG.parse(in_memory_file)
    
    contours = []
    
    for element in svg.elements():
        if isinstance(element, svgelements.Shape):
            path = svgelements.Path(element)
            for subpath in path.as_subpaths():
                points = []
                for seg in subpath:
                    if isinstance(seg, (svgelements.Move, svgelements.Close)):
                        if isinstance(seg, svgelements.Close) and len(points) > 0:
                            points.append(points[0])
                        continue
                    
                    seg_len = seg.length()
                    num_pts = max(2, int(seg_len / 1.0))
                    for i in range(num_pts + 1):
                        t = i / num_pts
                        pt = seg.point(t)
                        if not points or get_distance((points[-1][0], points[-1][1]), (pt.x, pt.y)) > 0.1:
                            points.append([pt.x, pt.y])
                if len(points) >= 2:
                    contours.append(np.array(points, dtype=np.float32).reshape(-1, 1, 2))
                    
    bbox = svg.bbox()
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0: width = 1000.0
        if height <= 0: height = 1000.0
    else:
        width, height = 1000.0, 1000.0

    return contours, float(width), float(height)


def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def contour_to_mm(contour: np.ndarray, config: dict) -> List[Tuple[float, float]]:
    mm_per_px_x = config.get("mm_per_px_x", 0.1)
    mm_per_px_y = config.get("mm_per_px_y", 0.1)
    offset_x_mm = config.get("offset_x_mm", 0.0)
    offset_y_mm = config.get("offset_y_mm", 0.0)
    points_mm = []
    for point in contour:
        x_px, y_px = point[0]
        points_mm.append((x_px * mm_per_px_x + offset_x_mm, y_px * mm_per_px_y + offset_y_mm))
    return points_mm


def point_line_distance(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    if start == end:
        return get_distance(point, start)
    px, py = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den if den > 0 else 0.0


def simplify_polyline(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points

    max_dist = 0.0
    split_idx = 0
    for i in range(1, len(points) - 1):
        dist = point_line_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            split_idx = i

    if max_dist > epsilon:
        left = simplify_polyline(points[: split_idx + 1], epsilon)
        right = simplify_polyline(points[split_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def fit_circle(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if len(points) < 3:
        return None

    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y

    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy, c0 = sol
    radius_sq = c0 + cx * cx + cy * cy
    if radius_sq <= 0:
        return None

    radius = math.sqrt(radius_sq)
    radial_errors = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)
    max_err = float(np.max(radial_errors))
    return float(cx), float(cy), float(radius), max_err


def try_fit_arc(
    points: List[Tuple[float, float]], tolerance_mm: float, min_sweep_rad: float
) -> Optional[Dict[str, float]]:
    circle = fit_circle(points)
    if circle is None:
        return None

    cx, cy, radius, max_err = circle
    if radius < 0.4 or max_err > tolerance_mm:
        return None

    pts = np.array(points, dtype=np.float64)
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    sweep = float(ang[-1] - ang[0])
    abs_sweep = abs(sweep)
    if abs_sweep < min_sweep_rad or abs_sweep > math.radians(330):
        return None

    clockwise = sweep < 0
    i_off = cx - points[0][0]
    j_off = cy - points[0][1]
    return {"clockwise": clockwise, "i": i_off, "j": j_off}


def fit_points_to_motions(points: List[Tuple[float, float]], config: dict) -> List[Tuple]:
    motions: List[Tuple] = []
    if len(points) < 2:
        return motions

    min_move_mm = config.get("min_move_mm", 0.08)
    arc_tol = config.get("arc_fit_tolerance_mm", 0.22)
    arc_min_pts = config.get("arc_fit_min_points", 6)
    arc_max_pts = config.get("arc_fit_max_points", 40)
    arc_min_sweep = math.radians(config.get("arc_fit_min_sweep_deg", 22))
    prioritize_arcs = config.get("prioritize_arcs", True)

    i = 0
    while i < len(points) - 1:
        if prioritize_arcs:
            best = None
            start_j = i + arc_min_pts - 1
            end_j = min(len(points) - 1, i + arc_max_pts - 1)
            if start_j <= end_j:
                for j in range(end_j, start_j - 1, -1):
                    fit = try_fit_arc(points[i : j + 1], arc_tol, arc_min_sweep)
                    if fit is not None:
                        best = (j, fit)
                        break
            if best is not None:
                j, fit = best
                x_end, y_end = points[j]
                if get_distance(points[i], (x_end, y_end)) >= min_move_mm:
                    motions.append(("G2" if fit["clockwise"] else "G3", x_end, y_end, fit["i"], fit["j"]))
                i = j
                continue

        x_end, y_end = points[i + 1]
        if get_distance(points[i], (x_end, y_end)) >= min_move_mm:
            motions.append(("G1", x_end, y_end))
        i += 1

    return motions


def merge_neighbor_lines_to_arcs(
    motions: List[Tuple], start_point: Tuple[float, float], config: dict
) -> List[Tuple]:
    if not motions:
        return motions

    merged: List[Tuple] = []
    cursor = start_point
    i = 0
    while i < len(motions):
        cmd = motions[i][0]
        if cmd != "G1":
            merged.append(motions[i])
            cursor = (motions[i][1], motions[i][2])
            i += 1
            continue

        run_points = [cursor]
        j = i
        while j < len(motions) and motions[j][0] == "G1":
            run_points.append((motions[j][1], motions[j][2]))
            j += 1

        if len(run_points) >= 5:
            merge_cfg = dict(config)
            merge_cfg["arc_fit_tolerance_mm"] = config.get("arc_fit_tolerance_mm", 0.22) * 1.22
            merge_cfg["arc_fit_min_sweep_deg"] = max(10, config.get("arc_fit_min_sweep_deg", 22) - 6)
            merged_run = fit_points_to_motions(run_points, merge_cfg)
            merged.extend(merged_run)
            cursor = run_points[-1]
        else:
            for k in range(1, len(run_points)):
                x_end, y_end = run_points[k]
                merged.append(("G1", x_end, y_end))
            cursor = run_points[-1]
        i = j

    return merged


def apply_arc_profile(config: dict, aggressiveness: int):
    effective_aggressiveness = float(aggressiveness) * 2.0
    n = max(0.0, min(2.0, effective_aggressiveness / 100.0))
    config["arc_fit_tolerance_mm"] = 0.10 + (0.28 - 0.10) * n
    config["arc_fit_min_points"] = max(3, int(round(8 - 4 * n)))
    config["arc_fit_max_points"] = int(round(28 + 44 * n))
    config["arc_fit_min_sweep_deg"] = max(4, 34 - (34 - 12) * n)
    config["rdp_epsilon_mm"] = 0.14 + (0.34 - 0.14) * n
    config["edge_post_merge_arcs"] = n >= 0.18


def polyline_to_motion_commands(points: List[Tuple[float, float]], config: dict) -> List[Tuple]:
    if len(points) < 2:
        return []

    epsilon = config.get("rdp_epsilon_mm", 0.35)
    simplified = simplify_polyline(points, epsilon)
    if len(simplified) < 2:
        return []
    motions = fit_points_to_motions(simplified, config)

    if (
        config.get("prioritize_arcs", True)
        and config.get("edge_post_merge_arcs", False)
        and config.get("run_mode") == "edge"
    ):
        motions = merge_neighbor_lines_to_arcs(motions, simplified[0], config)

    return motions


def optimize_gcode(gcode_lines: List[str]) -> List[str]:
    optimized = []
    pen_down = False
    last_motion = None

    for line in gcode_lines:
        cmd = line.strip()
        if not cmd:
            continue
        upper = cmd.upper()

        if upper == "M300 S30":
            if pen_down:
                continue
            pen_down = True
            optimized.append(cmd)
            continue
        if upper == "M300 S50":
            if not pen_down:
                continue
            pen_down = False
            optimized.append(cmd)
            continue

        if upper.startswith(("G0", "G1", "G2", "G3")):
            if cmd == last_motion:
                continue
            last_motion = cmd
            optimized.append(cmd)
            continue

        optimized.append(cmd)

    return optimized


def _gcode_xy(line: str, current_x: float, current_y: float) -> Tuple[float, float]:
    x = current_x
    y = current_y
    for part in line.split():
        if part.startswith("X"):
            x = float(part[1:])
        elif part.startswith("Y"):
            y = float(part[1:])
    return x, y


def optimize_gcode_travel_order(gcode_lines: List[str]) -> List[str]:
    header: List[str] = []
    footer: List[str] = []
    blocks: List[Dict[str, Any]] = []
    i = 0
    while i < len(gcode_lines):
        line = gcode_lines[i].strip()
        upper = line.upper()
        if upper.startswith("G0") and i + 1 < len(gcode_lines) and gcode_lines[i + 1].strip().upper() == "M300 S30":
            block = [line]
            start = _gcode_xy(line, 0.0, 0.0)
            end = start
            i += 1
            while i < len(gcode_lines):
                block_line = gcode_lines[i].strip()
                block.append(block_line)
                block_upper = block_line.upper()
                if block_upper.startswith(("G0", "G1", "G2", "G3")):
                    end = _gcode_xy(block_upper, end[0], end[1])
                i += 1
                if block_upper == "M300 S50":
                    break
            blocks.append({"start": start, "end": end, "lines": block})
            continue
        if blocks:
            footer.append(line)
        else:
            header.append(line)
        i += 1

    if len(blocks) < 2:
        return gcode_lines

    ordered = []
    cursor = (0.0, 0.0)
    unvisited = blocks[:]
    while unvisited:
        idx = min(
            range(len(unvisited)),
            key=lambda n: get_distance(cursor, unvisited[n]["start"]),
        )
        block = unvisited.pop(idx)
        ordered.extend(block["lines"])
        cursor = block["end"]
    return header + ordered + footer


def _gcode_draw_blocks(gcode_lines: List[str]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    header: List[str] = []
    footer: List[str] = []
    blocks: List[Dict[str, Any]] = []
    i = 0

    while i < len(gcode_lines):
        line = gcode_lines[i].strip()
        upper = line.upper()
        is_block_start = upper.startswith("G0") and i + 1 < len(gcode_lines) and gcode_lines[i + 1].strip().upper() == "M300 S30"
        if is_block_start:
            block = [line]
            start = _gcode_xy(line, 0.0, 0.0)
            cursor = start
            end = start
            draw_len = 0.0
            pen_down = False
            i += 1

            while i < len(gcode_lines):
                block_line = gcode_lines[i].strip()
                block.append(block_line)
                block_upper = block_line.upper()

                if block_upper == "M300 S30":
                    pen_down = True
                elif block_upper == "M300 S50":
                    pen_down = False
                elif block_upper.startswith(("G0", "G1", "G2", "G3")):
                    next_xy = _gcode_xy(block_upper, cursor[0], cursor[1])
                    if pen_down:
                        draw_len += get_distance(cursor, next_xy)
                    cursor = next_xy
                    end = next_xy

                i += 1
                if block_upper == "M300 S50":
                    break

            is_closed = draw_len > 0.0 and get_distance(start, end) <= max(2.5, draw_len * 0.08)
            blocks.append({"start": start, "end": end, "lines": block, "draw_len": draw_len, "closed": is_closed})
            continue

        if blocks:
            footer.append(line)
        else:
            header.append(line)
        i += 1

    return header, blocks, footer


def limit_gcode_line_budget(gcode_lines: List[str], target_lines: int) -> Tuple[List[str], int, bool]:
    original_count = len(gcode_lines)
    if target_lines <= 0 or original_count <= target_lines or target_lines >= 35000:
        return gcode_lines, original_count, False

    header, blocks, footer = _gcode_draw_blocks(gcode_lines)
    if len(blocks) < 2:
        return gcode_lines, original_count, False

    budget = max(20, target_lines - len(header) - len(footer))
    def score_block(block: Dict[str, Any]) -> float:
        draw_len = float(block.get("draw_len", 0.0))
        closed_bonus = 260.0 if block.get("closed") and 6.0 <= draw_len <= 320.0 else 0.0
        return draw_len + closed_bonus

    ranked = sorted(blocks, key=lambda block: (score_block(block), len(block.get("lines", []))), reverse=True)

    selected: List[Dict[str, Any]] = []
    used = 0
    for block in ranked:
        cost = len(block["lines"])
        if used + cost <= budget or not selected:
            selected.append(block)
            used += cost

    selected_ids = {id(block) for block in selected}
    kept_in_original_order = [block for block in blocks if id(block) in selected_ids]
    reduced = header + [line for block in kept_in_original_order for line in block["lines"]] + footer
    reduced = optimize_gcode_travel_order(reduced)
    return reduced, original_count, len(reduced) != original_count


def auto_tune_config(
    config: dict, image_gray: np.ndarray, run_mode: str, aspect_ratio: float, keep_user_offsets: bool = True
) -> dict:
    tuned = dict(config)
    contrast = float(np.std(image_gray))

    detail_boost = int(np.clip((contrast - 38.0) * 2.4, -90, 120))
    tuned["resolution_x"] = int(np.clip(tuned.get("resolution_x", 900) + detail_boost, 700, 1800))
    if tuned.get("link_resolution", True):
        tuned["resolution_y"] = int(np.clip(round(tuned["resolution_x"] * aspect_ratio), 200, 1800))
    else:
        tuned["resolution_y"] = int(np.clip(tuned.get("resolution_y", 900) + detail_boost, 200, 1800))
    tuned["min_line_length"] = 4 if contrast > 50 else 3

    if run_mode == "hatch":
        tuned["line_spacing_px"] = 2 if contrast > 52 else 3
    else:
        edge_for_stats = preprocess_for_edges(image_gray)
        low, high = estimate_canny_thresholds(edge_for_stats)
        tuned["canny_low"] = low
        tuned["canny_high"] = high
        tuned["edge_preprocess"] = True

    # Keep physical scale/offsets as chosen by user (legacy RU GUI behavior).
    tuned["width_mm"] = float(tuned.get("width_mm", 150.0))
    if keep_user_offsets:
        tuned["offset_x_mm"] = float(config.get("offset_x_mm", 0.0))
        tuned["offset_y_mm"] = float(config.get("offset_y_mm", 0.0))
    return tuned


def apply_line_budget_profile(config: dict):
    target = int(config.get("target_lines", 1000))
    pressure = float(np.clip((2200 - target) / 1900, 0.0, 1.0))
    if pressure <= 0:
        return

    config["min_line_length"] = max(int(config.get("min_line_length", 5)), int(round(5 + 18 * pressure)))
    config["rdp_epsilon_mm"] = max(float(config.get("rdp_epsilon_mm", 0.22)), 0.22 + 0.42 * pressure)
    config["min_move_mm"] = max(float(config.get("min_move_mm", 0.03)), 0.03 + 0.08 * pressure)
    config["arc_fit_tolerance_mm"] = max(float(config.get("arc_fit_tolerance_mm", 0.16)), 0.16 + 0.28 * pressure)
    config["arc_fit_min_sweep_deg"] = max(4, float(config.get("arc_fit_min_sweep_deg", 22)) - 8 * pressure)
    config["trace_binary_first"] = True
    config["contour_epsilon_ratio"] = max(float(config.get("contour_epsilon_ratio", 0.002)), 0.002 + 0.004 * pressure)
    config["min_contour_perimeter_px"] = max(float(config.get("min_contour_perimeter_px", 0.0)), 5.0 + 28.0 * pressure)
    config["min_contour_area_px"] = max(float(config.get("min_contour_area_px", 0.0)), 2.0 + 18.0 * pressure)


def extract_paths_from_skeleton(skeleton: np.ndarray) -> List[np.ndarray]:
    import networkx as nx
    
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return []

    G = nx.Graph()
    pixels = set(zip(ys, xs))
    
    for y, x in pixels:
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in pixels:
                    G.add_edge((y, x), (ny, nx_))
                    
    paths = []
    
    while G.number_of_edges() > 0:
        degrees = dict(G.degree())
        odd_nodes = [n for n, d in degrees.items() if d % 2 == 1 and d > 0]
        
        if odd_nodes:
            deg1 = [n for n in odd_nodes if degrees[n] == 1]
            start = deg1[0] if deg1 else odd_nodes[0]
        else:
            start = next(n for n, d in degrees.items() if d > 0)
            
        path = [start]
        curr = start
        
        while True:
            neighbors = list(G.neighbors(curr))
            if not neighbors:
                break
            
            next_node = neighbors[0]
            if len(neighbors) > 1 and len(path) > 1:
                prev = path[-2]
                dy, dx = curr[0] - prev[0], curr[1] - prev[1]
                best_n = neighbors[0]
                best_dot = -2.0
                for n in neighbors:
                    ndy, ndx = n[0] - curr[0], n[1] - curr[1]
                    n_len = (ndy**2 + ndx**2)**0.5
                    p_len = (dy**2 + dx**2)**0.5
                    if n_len > 0 and p_len > 0:
                        dot = (dy*ndy + dx*ndx) / (p_len * n_len)
                        if dot > best_dot:
                            best_dot = dot
                            best_n = n
                next_node = best_n
                
            G.remove_edge(curr, next_node)
            curr = next_node
            path.append(curr)
            
        if len(path) > 1:
            pts = np.array([[x, y] for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
            paths.append(pts)
            
    return paths


def trace_binary_contours(img: np.ndarray, config: dict) -> List[np.ndarray]:
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    try:
        thresh = threshold_otsu(denoised)
    except Exception:
        thresh = 128

    binary = (denoised < thresh).astype(np.uint8) * 255
    close_kernel = np.ones((2, 2), np.uint8)
    open_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

    from skimage.morphology import skeletonize
    skeleton = (skeletonize(binary > 0) * 255).astype(np.uint8)
    cv2.imwrite("debug_skeleton.png", skeleton)
    contours = extract_paths_from_skeleton(skeleton)
    min_perimeter = float(config.get("min_contour_perimeter_px", 6.0))
    epsilon_ratio = float(config.get("contour_epsilon_ratio", 0.002))

    filtered: List[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        if perimeter < min_perimeter:
            continue
        epsilon = max(0.7, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, False)
        if len(approx) >= 2:
            filtered.append(approx)

    filtered.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)
    return filtered


def dedupe_similar_contours(contours: List[np.ndarray], config: dict) -> List[np.ndarray]:
    dedupe_px = float(config.get("dedupe_contour_px", 4.0))
    kept: List[Tuple[np.ndarray, Tuple[float, float, float, float], float]] = []

    for contour in sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w * 0.5
        cy = y + h * 0.5
        length = cv2.arcLength(contour, False)
        duplicate = False
        for _, (kx, ky, kw, kh), klen in kept:
            kcx = kx + kw * 0.5
            kcy = ky + kh * 0.5
            if (
                abs(cx - kcx) <= dedupe_px
                and abs(cy - kcy) <= dedupe_px
                and abs(w - kw) <= dedupe_px * 2
                and abs(h - kh) <= dedupe_px * 2
                and min(length, klen) / max(length, klen, 1.0) > 0.72
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append((contour, (float(x), float(y), float(w), float(h)), length))

    return [item[0] for item in kept]


def generate_line_gcode(contours: List[np.ndarray], config: dict) -> List[str]:
    gcode = ["G21", "G90", "G28"]
    feedrate = config.get("feedrate", 2500)
    rapid_feedrate = config.get("rapid_feedrate", 4000)
    contours = dedupe_similar_contours(contours, config)

    unvisited = []
    for contour in contours:
        points_mm = contour_to_mm(contour, config)
        if len(points_mm) < 2:
            continue
        unvisited.append([points_mm[0], points_mm[-1], points_mm])

    sorted_contours = []
    current_pos = (0.0, 0.0)

    while unvisited:
        closest_idx = -1
        min_dist = float("inf")
        reverse = False

        for idx, (start_point, end_point, _) in enumerate(unvisited):
            dist_to_start = get_distance(current_pos, start_point)
            dist_to_end = get_distance(current_pos, end_point)

            if dist_to_start < min_dist:
                min_dist = dist_to_start
                closest_idx = idx
                reverse = False
            if dist_to_end < min_dist:
                min_dist = dist_to_end
                closest_idx = idx
                reverse = True

        start_point, end_point, points = unvisited.pop(closest_idx)
        if reverse:
            points = list(reversed(points))
            current_pos = start_point
        else:
            current_pos = end_point
        sorted_contours.append(points)

    for points in sorted_contours:
        x, y = points[0]
        gcode.append(f"G0 X{x:.2f} Y{y:.2f} F{rapid_feedrate}")
        gcode.append(config.get("pen_down", "M300 S30"))

        motions = polyline_to_motion_commands(points, config)
        for move in motions:
            if move[0] == "G1":
                _, mx, my = move
                gcode.append(f"G1 X{mx:.2f} Y{my:.2f} F{feedrate}")
            else:
                code, mx, my, i_off, j_off = move
                gcode.append(f"{code} X{mx:.2f} Y{my:.2f} I{i_off:.2f} J{j_off:.2f} F{feedrate}")
        gcode.append(config.get("pen_up", "M300 S50"))

    gcode.append("G28")
    return optimize_gcode(gcode)


def process_image_to_vectors(img: np.ndarray, config: Optional[dict] = None) -> Dict[str, Any]:
    return {"contours": trace_binary_contours(img, config or {})}


def process_image_to_edges(img: np.ndarray, config: dict, use_cuda: bool = False) -> Dict[str, Any]:
    if config.get("edge_preprocess", False):
        img = preprocess_for_edges(img)

    low_threshold = config.get("canny_low", 50)
    high_threshold = config.get("canny_high", 150)
    edges = None

    if use_cuda:
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            canny = cv2.cuda.createCannyEdgeDetector(low_threshold, high_threshold)
            edges = canny.detect(gpu_img).download()
        except Exception:
            edges = None

    if edges is None:
        edges = cv2.Canny(img, low_threshold, high_threshold)
        
    from skimage.morphology import skeletonize
    skel = skeletonize(edges > 0)
    contours = extract_paths_from_skeleton(skel)

    simplified_contours = []
    min_len = 2  # Keep small segments
    min_perimeter = float(config.get("min_contour_perimeter_px", 0.0))
    min_area = float(config.get("min_contour_area_px", 0.0))
    epsilon_ratio = float(config.get("contour_epsilon_ratio", 0.002))
    for contour in contours:
        if len(contour) < min_len:
            continue
        perimeter = cv2.arcLength(contour, False)
        area = abs(cv2.contourArea(contour))
        if perimeter < min_perimeter and area < min_area:
            continue
        epsilon = max(0.5, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, False)
        if len(approx) >= 2:
            simplified_contours.append(approx)

    return {"contours": simplified_contours}


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
    return apply_floyd_steinberg_dither(img_float)


def generate_hatching_gcode(dithered_img: np.ndarray, config: dict) -> List[str]:
    gcode = ["G21", "G90", "G28"]
    feedrate = config.get("feedrate", 2500)
    rapid_feedrate = config.get("rapid_feedrate", 4000)
    line_spacing_px = config.get("line_spacing_px", 3)
    mm_per_px_x = config.get("mm_per_px_x", 0.1)
    mm_per_px_y = config.get("mm_per_px_y", 0.1)
    offset_x_mm = config.get("offset_x_mm", 0.0)
    offset_y_mm = config.get("offset_y_mm", 0.0)

    height, width = dithered_img.shape
    gcode.append(config.get("pen_up", "M300 S50"))
    pen_is_down = False

    for y_px in range(0, height, line_spacing_px):
        y_mm = y_px * mm_per_px_y + offset_y_mm
        x_range = range(width) if (y_px // line_spacing_px) % 2 == 0 else range(width - 1, -1, -1)

        if pen_is_down:
            gcode.append(config.get("pen_up", "M300 S50"))
            pen_is_down = False

        for x_px in x_range:
            pixel_brightness = dithered_img[y_px, x_px]
            x_mm = x_px * mm_per_px_x + offset_x_mm

            if pixel_brightness < 0.5:
                if not pen_is_down:
                    gcode.append(f"G0 X{x_mm:.2f} Y{y_mm:.2f} F{rapid_feedrate}")
                    gcode.append(config.get("pen_down", "M300 S30"))
                    pen_is_down = True
                else:
                    gcode.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feedrate}")
            else:
                if pen_is_down:
                    gcode.append(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feedrate}")
                    gcode.append(config.get("pen_up", "M300 S50"))
                    pen_is_down = False

    if pen_is_down:
        gcode.append(config.get("pen_up", "M300 S50"))
    gcode.append("G28")
    return optimize_gcode(gcode)


def create_gcode_visualization(gcode_lines: List[str]) -> str:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    current_x, current_y = 0.0, 0.0
    pen_down = False
    drawn_x: List[float] = []
    drawn_y: List[float] = []
    travel_x: List[float] = []
    travel_y: List[float] = []

    for line in gcode_lines:
        line = line.strip().upper()
        if not line or line.startswith(";"):
            continue
        if line == "M300 S30":
            pen_down = True
            continue
        if line == "M300 S50":
            pen_down = False
            continue
        if line.startswith(("G0", "G1")):
            parts = line.split()
            new_x, new_y = current_x, current_y
            for part in parts:
                if part.startswith("X"):
                    new_x = float(part[1:])
                elif part.startswith("Y"):
                    new_y = float(part[1:])

            color = "black" if pen_down else "red"
            linewidth = 1.0 if pen_down else 0.55
            alpha = 1.0 if pen_down else 0.55
            ax.plot([current_x, new_x], [current_y, new_y], color=color, linewidth=linewidth, alpha=alpha)
            if pen_down:
                drawn_x.extend([current_x, new_x])
                drawn_y.extend([current_y, new_y])
            else:
                travel_x.extend([current_x, new_x])
                travel_y.extend([current_y, new_y])
            current_x, current_y = new_x, new_y
        elif line.startswith(("G2", "G3")):
            parts = line.split()
            x_end, y_end = current_x, current_y
            i_off, j_off = 0.0, 0.0
            for part in parts:
                if part.startswith("X"):
                    x_end = float(part[1:])
                elif part.startswith("Y"):
                    y_end = float(part[1:])
                elif part.startswith("I"):
                    i_off = float(part[1:])
                elif part.startswith("J"):
                    j_off = float(part[1:])

            cx = current_x + i_off
            cy = current_y + j_off
            r = math.hypot(current_x - cx, current_y - cy)
            if r > 1e-6:
                a0 = math.atan2(current_y - cy, current_x - cx)
                a1 = math.atan2(y_end - cy, x_end - cx)
                sweep = a1 - a0
                clockwise = line.startswith("G2")
                if clockwise and sweep >= 0:
                    sweep -= 2 * math.pi
                elif not clockwise and sweep <= 0:
                    sweep += 2 * math.pi

                steps = max(8, int(abs(sweep) * 24))
                angles = np.linspace(a0, a0 + sweep, steps)
                xs = cx + r * np.cos(angles)
                ys = cy + r * np.sin(angles)
                color = "black" if pen_down else "red"
                linewidth = 1.0 if pen_down else 0.55
                alpha = 1.0 if pen_down else 0.55
                ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
                if pen_down:
                    drawn_x.extend(xs.tolist())
                    drawn_y.extend(ys.tolist())
                else:
                    travel_x.extend(xs.tolist())
                    travel_y.extend(ys.tolist())
            current_x, current_y = x_end, y_end

    if drawn_x and drawn_y:
        # Keep zoom focused on drawn content, with travel lines still visible where possible.
        x_min = min(drawn_x)
        x_max = max(drawn_x)
        y_min = min(drawn_y)
        y_max = max(drawn_y)
        if travel_x and travel_y:
            x_min = min(x_min, min(travel_x))
            x_max = max(x_max, max(travel_x))
            y_min = min(y_min, min(travel_y))
            y_max = max(y_max, max(travel_y))
        pad = max(3.0, 0.05 * max(x_max - x_min, y_max - y_min))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title("G-code Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def send_gcode_to_ev3(commands: List[str], host: str, port: int = 15614) -> str:
    if not commands:
        raise ValueError("G-code is empty")

    payload = {"version": 1, "commands": commands}
    json_data = json.dumps(payload, separators=(",", ":"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(60.0)
        s.connect((host, port))
        s.sendall(json_data.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        response = s.recv(1024).decode("utf-8")
        if response != "OK":
            raise RuntimeError(f"EV3 returned error: {response}")

    return "OK"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image_route():
    try:
        if "image_file" not in request.files:
            return jsonify({"error": "Image file not found"}), 400

        file = request.files["image_file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        config = {
            "feedrate": 2500,
            "rapid_feedrate": 4000,
            "pen_down": "M300 S30",
            "pen_up": "M300 S50",
            "canny_low": int(request.form.get("canny_low", 50)),
            "canny_high": int(request.form.get("canny_high", 150)),
            "line_spacing_px": int(request.form.get("spacing_px", 3)),
            "min_line_length": 5,
            "width_mm": float(np.clip(float(request.form.get("width_mm", 900.0)), 20.0, 900.0)),
            "offset_x_mm": float(np.clip(float(request.form.get("offset_x_mm", 0.0)), -100.0, 100.0)),
            "offset_y_mm": float(np.clip(float(request.form.get("offset_y_mm", 0.0)), -100.0, 100.0)),
            "resolution_x": int(np.clip(int(request.form.get("resolution_x", 900)), 200, 1800)),
            "resolution_y": int(np.clip(int(request.form.get("resolution_y", 900)), 200, 1800)),
            "target_lines": int(np.clip(int(request.form.get("target_lines", 35000)), 200, 35000)),
            "link_resolution": request.form.get("link_resolution", "on") == "on",
            "prioritize_arcs": request.form.get("prioritize_arcs", "on") == "on",
            "arc_aggressiveness": int(request.form.get("arc_aggressiveness", 65)),
            "rdp_epsilon_mm": 0.22,
            "min_move_mm": 0.03,
            "arc_fit_tolerance_mm": 0.16,
            "arc_fit_min_points": 6,
            "arc_fit_max_points": 40,
            "arc_fit_min_sweep_deg": 22,
            "edge_preprocess": False,
            "edge_post_merge_arcs": True,
        }

        run_mode = request.form.get("mode")
        is_svg = file.filename.lower().endswith(".svg")
        if run_mode is None:
            run_mode = "vector"

        auto_optimize = request.form.get("auto_optimize", "off") == "on"
        use_cuda = has_cuda_acceleration()
        apply_arc_profile(config, config["arc_aggressiveness"])

        if is_svg:
            file.seek(0)
            svg_contours, svg_w, svg_h = parse_svg_to_contours(file)
            aspect_ratio = svg_h / svg_w if svg_w > 0 else 1.0
            original_img_gray = None
        else:
            file.seek(0)
            original_img_gray = load_image_grayscale(file.filename, file_storage=file)
            original_height, original_width = original_img_gray.shape
            aspect_ratio = original_height / original_width

        if config["link_resolution"]:
            config["resolution_y"] = int(np.clip(round(config["resolution_x"] * aspect_ratio), 200, 1800))

        if auto_optimize:
            if original_img_gray is not None:
                config = auto_tune_config(config, original_img_gray, run_mode, aspect_ratio, keep_user_offsets=True)
            apply_arc_profile(config, config["arc_aggressiveness"])
            apply_line_budget_profile(config)

        config["run_mode"] = run_mode

        resolution_x = int(np.clip(config.get("resolution_x", 900), 200, 1800))
        resolution_y = int(np.clip(config.get("resolution_y", 900), 200, 1800))
        output_size_px = (resolution_x, resolution_y)
        height_mm = config["width_mm"] * (resolution_y / resolution_x)
        config["mm_per_px_x"] = config["width_mm"] / resolution_x
        config["mm_per_px_y"] = height_mm / resolution_y

        if is_svg:
            scaled_contours = []
            scale_x = resolution_x / svg_w if svg_w > 0 else 1.0
            scale_y = resolution_y / svg_h if svg_h > 0 else 1.0
            for c in svg_contours:
                sc = c.copy()
                sc[:, 0, 0] *= scale_x
                sc[:, 0, 1] *= scale_y
                scaled_contours.append(sc)
            gcode_lines = generate_line_gcode(scaled_contours, config)
        else:
            resized_img = resize_image(original_img_gray, output_size_px, use_cuda=use_cuda)

            if run_mode == "hatch":
                dithered_img = process_image_for_hatching(resized_img)
                gcode_lines = generate_hatching_gcode(dithered_img, config)
            elif run_mode == "vector":
                result = process_image_to_vectors(resized_img, config)
                gcode_lines = generate_line_gcode(result["contours"], config)
            elif config.get("trace_binary_first", False):
                result = process_image_to_vectors(resized_img, config)
                gcode_lines = generate_line_gcode(result["contours"], config)
            else:
                result = process_image_to_edges(resized_img, config, use_cuda=use_cuda)
                gcode_lines = generate_line_gcode(result["contours"], config)

        if not gcode_lines:
            return jsonify({"error": "Failed to generate G-code"}), 400

        gcode_lines = optimize_gcode_travel_order(gcode_lines)
        original_line_count = len(gcode_lines)
        line_budget_applied = False
        if auto_optimize:
            gcode_lines, original_line_count, line_budget_applied = limit_gcode_line_budget(
                gcode_lines, config["target_lines"]
            )
        preview_image_base64 = create_gcode_visualization(gcode_lines)
        return jsonify(
            {
                "message": "Processed successfully",
                "gcode_lines": gcode_lines,
                "line_count": len(gcode_lines),
                "original_line_count": original_line_count,
                "line_budget_applied": line_budget_applied,
                "target_lines": config["target_lines"],
                "preview_image": preview_image_base64,
                "auto_optimized": auto_optimize,
                "cuda_used": use_cuda,
                "prioritize_arcs": config["prioritize_arcs"],
                "arc_aggressiveness": config["arc_aggressiveness"],
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "width_mm": float(config["width_mm"]),
                "offset_x_mm": float(config["offset_x_mm"]),
                "offset_y_mm": float(config["offset_y_mm"]),
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def send_gcode_task(commands: List[str], host: str, port: int):
    try:
        send_gcode_to_ev3(commands, host, port)
    except Exception:
        pass


@app.route("/send_to_ev3", methods=["POST"])
def send_to_ev3_route():
    try:
        data = request.json
        commands = data.get("commands")
        host = data.get("host")
        port = int(data.get("port", 15614))

        if not commands or not host:
            return jsonify({"error": "Missing G-code or IP"}), 400

        thread = threading.Thread(target=send_gcode_task, args=(commands, host, port), daemon=True)
        thread.start()

        return jsonify({"message": "Send initiated"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/preview_gcode", methods=["POST"])
def preview_gcode_route():
    try:
        data = request.json or {}
        commands = data.get("commands", [])
        if not isinstance(commands, list) or not commands:
            return jsonify({"error": "Missing G-code commands"}), 400
        preview_image_base64 = create_gcode_visualization(commands)
        return jsonify({"preview_image": preview_image_base64, "line_count": len(commands)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Plott3r PC app running: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)

import json
import base64
import io
from PIL import Image
import numpy as np
import cv2
import main

def process_image(image_bytes, mode, config_json):
    # Decode image from bytes
    data = bytes(image_bytes)
    try:
        from android.graphics import BitmapFactory, Bitmap
        from java.io import ByteArrayOutputStream
        bitmap = BitmapFactory.decodeByteArray(data, 0, len(data))
        out = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        nparr = np.frombuffer(bytes(out.toByteArray()), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    except Exception:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
    if img is None:
        return json.dumps({"error": "Cannot decode image"})
        
    # Read config from JSON
    config = json.loads(config_json)
    
    # Process
    if mode == "edge":
        result = main.process_image_to_edges(img, config, False)
        contours = result["contours"]
        paths_mm = main.optimize_and_scale_paths(contours, config, "edge")
        gcode = main.generate_gcode(paths_mm, config)
        
        # Render preview
        preview_img = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        for cnt in contours:
            pts = cnt.astype(np.int32)
            for i in range(len(pts)-1):
                cv2.line(preview_img, tuple(pts[i][0]), tuple(pts[i+1][0]), 0, 1)
        
    elif mode == "hatching":
        dithered = main.process_image_for_hatching(img)
        gcode = main.generate_hatching_gcode(dithered, config)
        
        preview_img = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
        preview_img[dithered < 0.5] = 0
    else:
        return json.dumps({"error": "Unknown mode"})
        
    # Convert preview to base64 jpg for Android to render easily
    success, buffer = cv2.imencode(".jpg", preview_img)
    if not success:
        return json.dumps({"error": "Failed to encode preview"})
    preview_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return json.dumps({
        "success": True,
        "gcode": gcode,
        "preview_b64": preview_b64,
        "lines": len(gcode)
    })

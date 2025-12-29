# <copy this entire file to object_detection.py> 

"""
object_detection.py

Drop-in, robust YOLOv3-tiny detection for your project.
Defaults match your repo structure:
 - cfg:  object_detection_model/config/yolov3-tiny.cfg
 - weights: object_detection_model/weights/yolov3-tiny.weights
 - names: object_detection_model/objectLabels/coco.names

Exposes:
 - get_detected_objects(frame, conf_threshold=0.25)
 - get_detected_labels(frame, conf_threshold=0.25, unique=True)

Usage (examples):
  python object_detection.py                 # open webcam and show detections
  python object_detection.py path/to/img.jpg # detect on single image and show
  # from another script:
  from object_detection import get_detected_labels
  labels = get_detected_labels(frame)
"""
import cv2
import numpy as np
import time
import sys
import os

# --- Configuration (change only if you moved files) ---
WEIGHTS_PATH = "object_detection_model/weights/yolov3-tiny.weights"
CFG_PATH     = "object_detection_model/config/yolov3-tiny.cfg"
NAMES_PATH   = "object_detection_model/objectLabels/coco.names"

# default input source: 0 => webcam. You can override via command-line.
INPUT_SOURCE = 0
# ----------------------------------------------------

# Load network
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
if net.empty():
    raise RuntimeError("Failed to load network. Check WEIGHTS_PATH and CFG_PATH")

# Load class names
if not os.path.exists(NAMES_PATH):
    raise FileNotFoundError(f"Names file not found: {NAMES_PATH}")
with open(NAMES_PATH, "r") as f:
    label_classes = [c.strip() for c in f.readlines() if c.strip()]

print(f"[INFO] Loaded {len(label_classes)} class names.")

# Resolve output layer names robustly
try:
    output_layers = net.getUnconnectedOutLayersNames()
    print("[INFO] Using getUnconnectedOutLayersNames():", output_layers)
except Exception:
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    try:
        flat = np.array(unconnected).flatten().astype(int)
        output_layers = [layer_names[i - 1] for i in flat]
    except Exception:
        output_layers = []
        for ii in unconnected:
            try:
                idx = int(ii)
            except Exception:
                idx = int(ii[0])
            output_layers.append(layer_names[idx - 1])
    print("[INFO] Using fallback output_layers:", output_layers)

colors = np.random.uniform(0, 255, size=(len(label_classes), 3)).astype(int).tolist()
font = cv2.FONT_HERSHEY_SIMPLEX

def detectObject(frame,
                 conf_threshold=0.25,
                 nms_threshold=0.4,
                 inp_width=416,
                 inp_height=416,
                 draw=True):
    """
    Low-level detector: runs forward pass on `frame` which is expected to be
    in the same coordinate space as inp_width x inp_height.
    Returns list of detections: (label, confidence, (x, y, w, h))
    Coordinates are relative to the input `frame` passed here.
    If draw=True, draws rectangles on the provided frame in-place.
    """
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0,
                                 size=(inp_width, inp_height),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        if out is None:
            continue
        for detection in out:
            if len(detection) < 6:
                continue
            objectness = float(detection[4])
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            confidence = objectness * class_score

            if confidence > conf_threshold:
                cx = int(detection[0] * W)
                cy = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    detections = []
    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        flat = []
        if isinstance(indexes, (list, tuple)):
            for it in indexes:
                if isinstance(it, (list, tuple, np.ndarray)):
                    try:
                        flat.append(int(it[0]))
                    except:
                        pass
                else:
                    try:
                        flat.append(int(it))
                    except:
                        pass
        elif isinstance(indexes, np.ndarray):
            try:
                flat = indexes.flatten().astype(int).tolist()
            except:
                flat = [int(x) for x in indexes]
        else:
            try:
                flat = [int(indexes)]
            except:
                flat = []

        for i in flat:
            if i < 0 or i >= len(boxes):
                continue
            x, y, w, h = boxes[i]
            cls_id = class_ids[i] if i < len(class_ids) else -1
            label = label_classes[cls_id] if (0 <= cls_id < len(label_classes)) else str(cls_id)
            conf = confidences[i]
            detections.append((label, conf, (x, y, w, h)))

            if draw:
                color = colors[cls_id % len(colors)] if cls_id >= 0 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{label}: {conf:.2f}"
                ty = max(y - 6, 12)
                cv2.putText(frame, text, (x, ty), font, 0.5, color, 1, cv2.LINE_AA)

    return detections

def _open_source(src):
    """Return tuple (type, obj). type: 'image'|'video' """
    if isinstance(src, str) and os.path.isfile(src):
        ext = os.path.splitext(src)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            img = cv2.imread(src)
            if img is None:
                raise RuntimeError("Failed to read image")
            return "image", img
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {src}")
        return "video", cap
    # try webcam index
    cap = cv2.VideoCapture(int(src))
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam/camera index.")
    return "video", cap

def letterbox_image(img, target_w, target_h, color=(114,114,114)):
    """
    Resize image to fit into (target_w, target_h) while keeping aspect ratio.
    Pads the rest with `color`.
    Returns:
      - letterboxed image (target_h, target_w, 3)
      - scale (float) used to scale original -> new
      - pad_left, pad_top (ints) applied to center the image
    """
    h, w = img.shape[:2]
    # compute scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    # pad with color
    letterbox = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=color)
    return letterbox, scale, pad_left, pad_top

# -------- New reusable helpers --------
def get_detected_objects(frame, conf_threshold=0.25, inp_w=416, inp_h=416):
    """
    Run detection on an arbitrary BGR frame and return detections mapped to original coords.
    Returns: list of tuples (label, confidence, (x, y, w, h)) in original frame coordinates.
    Does NOT draw on the frame.
    """
    orig_h, orig_w = frame.shape[:2]
    # letterbox to preserve aspect ratio
    resized, scale, pad_left, pad_top = letterbox_image(frame, inp_w, inp_h)
    det_resized = detectObject(resized, conf_threshold=conf_threshold,
                               nms_threshold=0.4, inp_width=inp_w, inp_height=inp_h, draw=False)

    detections = []
    for (label, conf, (x, y, w, h)) in det_resized:
        x_unpad = x - pad_left
        y_unpad = y - pad_top

        x0 = int(round(x_unpad / scale))
        y0 = int(round(y_unpad / scale))
        w0 = int(round(w / scale))
        h0 = int(round(h / scale))

        x0 = max(0, min(x0, orig_w - 1))
        y0 = max(0, min(y0, orig_h - 1))
        w0 = max(0, min(w0, orig_w - x0))
        h0 = max(0, min(h0, orig_h - y0))

        detections.append((label, conf, (x0, y0, w0, h0)))

    return detections

def get_detected_labels(frame, conf_threshold=0.25, inp_w=416, inp_h=416, unique=True):
    """
    Convenience helper — returns a list of detected label strings for the frame.
    If unique=True, returns unique labels preserving first-seen order.
    """
    dets = get_detected_objects(frame, conf_threshold=conf_threshold, inp_w=inp_w, inp_h=inp_h)
    labels = [d[0] for d in dets]
    if unique:
        seen = set()
        uniq = []
        for l in labels:
            if l not in seen:
                uniq.append(l); seen.add(l)
        return uniq
    return labels

# ---------------------------------------

def main():
    # allow override via CLI
    src = INPUT_SOURCE
    if len(sys.argv) > 1:
        src = sys.argv[1]

    # determine blob size from cfg content if possible
    inp_w = 416
    inp_h = 416
    try:
        with open(CFG_PATH, "r") as f:
            header = f.read(1024).lower()
            if "width=608" in header or "height=608" in header:
                inp_w = inp_h = 608
    except:
        pass

    print(f"[INFO] Using blob size: {inp_w}x{inp_h}")
    src_type, src_obj = _open_source(src)
    print(f"[INFO] Source type: {src_type}")

    if src_type == "image":
        frame = src_obj
        t0 = time.time()
        detections = get_detected_objects(frame, conf_threshold=0.25, inp_w=inp_w, inp_h=inp_h)
        t1 = time.time()
        print("[INFO] Detections:", detections)
        print(f"[INFO] Inference time: {t1-t0:.3f}s")
        # draw
        for (label, conf, (x, y, w, h)) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (x, max(y - 6, 12)),
                        font, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print just labels (unique order):
        labels = get_detected_labels(frame, conf_threshold=0.25, inp_w=inp_w, inp_h=inp_h, unique=True)
        print("[INFO] Labels:", labels)
        return labels

    cap = src_obj
    frame_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or failed to read frame.")
                break

            # get detections mapped to original frame coords
            detections = get_detected_objects(frame, conf_threshold=0.25, inp_w=inp_w, inp_h=inp_h)

            # draw detections
            for (label, conf, (x, y, w, h)) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.putText(frame, f"{label}:{conf:.2f}", (x, max(y - 6, 12)),
                            font, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # prepare label list (unique)
            labels = get_detected_labels(frame, conf_threshold=0.25, inp_w=inp_w, inp_h=inp_h, unique=True)
            if labels:
                summary = ", ".join([f"{l}" for l in labels[:10]])
                cv2.putText(frame, summary, (5, 20), font, 0.5, (0,255,0), 1)

            cv2.imshow("YOLO (letterbox)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Stopped by user (q).")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time + 1e-6)
                print(f"[INFO] FPS={fps:.2f}, detections={len(detections)}, labels={labels}")

    finally:
        if isinstance(cap, cv2.VideoCapture):
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

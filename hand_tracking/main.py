import cv2
import numpy as np
import mediapipe as mp
import math
import time
import os

MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy '{MODEL_PATH}'")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
W, H = 640, 480

SMOOTH_FACTOR = 0.6
MAX_MISSED_FRAMES = 2


EFFECT_NAMES = [
    "Rainbow", "Invert", "Pure B&W", "Blur", "Pixelate",
    "Invert Red", "Invert Green", "Invert Blue",
    "Sketch", "Edge", "Thermal", "Night Vision"
]


SHAPE_NAMES = ["cube", "pyramid", "cylinder"]
SPIN_SPEED = (0.04, 0.025, 0.015)

V_CUBE = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=np.float32)
F_CUBE = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

V_PYR = np.array([[0,-1.5,0],[-1,1,-1],[1,1,-1],[0,1,1]], dtype=np.float32)
F_PYR = [[0,1,2],[0,2,3],[0,3,1],[1,2,3]]

def make_cylinder(slices=8):
    verts = []
    top, bottom = [], []
    for i in range(slices):
        a = i * 2 * math.pi / slices
        verts.append([math.cos(a), 1.2, math.sin(a)])
        verts.append([math.cos(a), -1.2, math.sin(a)])
        top.append(i*2)
        bottom.append(i*2+1)
    faces = [top, bottom]
    for i in range(slices):
        nxt = (i+1) % slices
        faces.append([i*2, nxt*2, nxt*2+1, i*2+1])
    return np.array(verts, dtype=np.float32), faces

V_CYL, F_CYL = make_cylinder()
SHAPES = [(V_CUBE, F_CUBE), (V_PYR, F_PYR), (V_CYL, F_CYL)]


def get_palm_width(hand):
    return math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y)

def is_fist(hand, palm_width):
    
    threshold = palm_width * 0.85
    for tip, mcp in [(8,5),(12,9),(16,13),(20,17)]:
        if math.hypot(hand[tip].x - hand[mcp].x, hand[tip].y - hand[mcp].y) > threshold:
            return False
    return True

def apply_effect(frame, mode):
    if mode == 0:   
        return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
    if mode == 1:   
        return cv2.bitwise_not(frame)
    if mode == 2:   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if mode == 3:   
        return cv2.GaussianBlur(frame, (91, 91), 0)
    if mode == 4:   
        small = cv2.resize(frame, (W//35, H//35), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    if mode == 5:   
        b, g, r = cv2.split(frame)
        inv_r = 255 - r
        return cv2.merge([b, g, inv_r])
    if mode == 6:   
        b, g, r = cv2.split(frame)
        inv_g = 255 - g
        return cv2.merge([b, inv_g, r])
    if mode == 7:   
        b, g, r = cv2.split(frame)
        inv_b = 255 - b
        return cv2.merge([inv_b, g, r])
    if mode == 8:   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        return cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if mode == 9:   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if mode == 10:  
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    if mode == 11:  
        green = np.zeros_like(frame)
        green[:,:,1] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return green
    return frame


class HologramEngine:
    def __init__(self):
        self.angle_x = self.angle_y = self.angle_z = 0.0
        self.shape_idx = 0
        self.was_fist = False

    def process(self, hand, frame):
        h, w = frame.shape[:2]

        
        palm_center = hand[9]
        cx = int(palm_center.x * w)
        cy = int(palm_center.y * h)

        palm_width = get_palm_width(hand) * w
        scale = int(palm_width * 0.4)

        
        fist_now = is_fist(hand, palm_width)
        if fist_now:
            self.was_fist = True
        elif self.was_fist and not fist_now:
            self.shape_idx = (self.shape_idx + 1) % len(SHAPES)
            self.was_fist = False

        
        self.angle_x += SPIN_SPEED[0]
        self.angle_y += SPIN_SPEED[1]
        self.angle_z += SPIN_SPEED[2]

        verts, faces = SHAPES[self.shape_idx]

        
        cx_a, sx_a = math.cos(self.angle_x), math.sin(self.angle_x)
        cy_a, sy_a = math.cos(self.angle_y), math.sin(self.angle_y)
        cz_a, sz_a = math.cos(self.angle_z), math.sin(self.angle_z)

        y1 = verts[:,1]*cx_a - verts[:,2]*sx_a
        z1 = verts[:,1]*sx_a + verts[:,2]*cx_a
        x1 = verts[:,0]

        x2 = x1*cy_a + z1*sy_a
        z2 = -x1*sy_a + z1*cy_a
        y2 = y1

        x3 = x2*cz_a - y2*sz_a
        y3 = x2*sz_a + y2*cz_a

        pts_2d = np.column_stack((x3*scale + cx, y3*scale + cy)).astype(np.int32)

        
        face_depths = []
        for face in faces:
            z_avg = sum(z2[idx] for idx in face) / len(face)
            poly = pts_2d[face]
            face_depths.append((z_avg, poly))

        
        face_depths.sort(key=lambda x: x[0], reverse=True)

        
        mask_3d = np.zeros((h, w), dtype=np.uint8)
        for _, poly in face_depths:
            cv2.fillPoly(mask_3d, [poly], 255)

        
        mask_3f = mask_3d[:,:,None] / 255.0
        invert = cv2.bitwise_not(frame)
        frame = (frame * (1 - mask_3f) + invert * mask_3f).astype(np.uint8)

        
        wire_canvas = np.zeros_like(frame)
        for _, poly in face_depths:
            cv2.polylines(wire_canvas, [poly], True, (0, 255, 255), 2)
        frame = cv2.add(frame, wire_canvas)

        return frame


class HandTracker:
    def __init__(self):
        self.smoothed_pts = None
        self.missed_frames = 0
        self.is_pinched = False
        self.effect_mode = 0

    def update(self, current_pts):
        if current_pts is not None:
            if self.smoothed_pts is None:
                self.smoothed_pts = current_pts
            else:
                self.smoothed_pts = [
                    (int(self.smoothed_pts[i][0]*(1-SMOOTH_FACTOR) + current_pts[i][0]*SMOOTH_FACTOR),
                     int(self.smoothed_pts[i][1]*(1-SMOOTH_FACTOR) + current_pts[i][1]*SMOOTH_FACTOR))
                    for i in range(4)
                ]
            self.missed_frames = 0
        else:
            self.missed_frames += 1
            if self.missed_frames > MAX_MISSED_FRAMES:
                self.smoothed_pts = None

    def check_pinch(self, threshold=120):
        if self.smoothed_pts is None: return
        max_dist = max(math.hypot(self.smoothed_pts[i][0]-self.smoothed_pts[j][0],
                                  self.smoothed_pts[i][1]-self.smoothed_pts[j][1])
                       for i in range(4) for j in range(i+1,4))
        if max_dist < threshold:
            if not self.is_pinched:
                self.effect_mode = (self.effect_mode + 1) % len(EFFECT_NAMES)
                self.is_pinched = True
        else:
            self.is_pinched = False


def main():
    tracker = HandTracker()
    hologram = HologramEngine()
    prev_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            timestamp_ms = int(time.time_ns() // 1_000_000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            num_hands = len(results.handedness) if results.hand_landmarks else 0

            
            if num_hands == 1:
                hand = results.hand_landmarks[0]
                frame = hologram.process(hand, frame)
                tracker.smoothed_pts = None

            
            elif num_hands == 2:
                pts_left = pts_right = None
                for idx, handedness in enumerate(results.handedness):
                    label = handedness[0].category_name
                    hand = results.hand_landmarks[idx]
                    thumb = (int(hand[4].x*W), int(hand[4].y*H))
                    index = (int(hand[8].x*W), int(hand[8].y*H))
                    if label == 'Left':
                        pts_left = (thumb, index)
                    else:
                        pts_right = (thumb, index)

                if pts_left and pts_right:
                    current_pts = [pts_left[0], pts_left[1], pts_right[1], pts_right[0]]
                else:
                    current_pts = None

                tracker.update(current_pts)

                if tracker.smoothed_pts is not None:
                    tracker.check_pinch(120)

                    effect_frame = apply_effect(frame, tracker.effect_mode)
                    quad = np.array(tracker.smoothed_pts, np.int32)
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [quad], 255)
                    mask_3f = mask[:,:,None] / 255.0
                    frame = (frame * (1-mask_3f) + effect_frame * mask_3f).astype(np.uint8)

                    cv2.polylines(frame, [quad], True, (255,255,255), 3)
                    for pt in tracker.smoothed_pts:
                        cv2.circle(frame, pt, 6, (0,0,255), -1)

                else:
                    tracker.smoothed_pts = None

            
            curr_time = time.time()
            prev_time = curr_time

            cv2.imshow("trend test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
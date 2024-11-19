from collections.abc import Generator
import numpy as np
import cv2

def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Generator[np.ndarray, None, None]:
    def calc(t):
        return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3
    # get the approximate pixel count of the curve
    approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
    for t in np.linspace(0, 1, round(approx * 1.2)):
        yield np.round(calc(t)).astype(np.int32)
        
def generate_scratch(img: np.ndarray, max_length: float, end_brush_range: tuple[float, float], mid_brush_range: tuple[float, float]) -> np.ndarray:
    H, W = img.shape
    # generate the 2 end points of the bezier curve
    x, y, rho1, theta1 = np.random.uniform([0] * 4, [W, H, max_length, np.pi * 2])
    p1 = np.array([x, y, 0])
    p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]
    # generate the second point, make sure that it cannot be too far away from the middle point of the 2 end points
    rho2, theta2 = np.random.uniform([0], [rho1 / 2, np.pi * 2])
    p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]
    # generate the brush sizes of the 3 points
    p1[2], p2[2], p3[2] = np.random.uniform(*np.transpose([end_brush_range, mid_brush_range, end_brush_range]))
    for x, y, brush in bezier(p1, p2, p3):
        cv2.circle(img, (x, y), brush, 255, -1)
    return img

def create_stroke_prior(mask, exclude_random_class=False):
    MAX_LENGTH = 512  # maximum distance between two end points
    END_BRUSH_RANGE = (np.random.randint(2,15), np.random.randint(2,15))  # brush size range of the two end points
    MID_BRUSH_RANGE = (np.random.randint(2,15), np.random.randint(2,15))  # brush size range of the mid point
    SCRATCH_CNT = 20
    strokes = np.zeros_like(mask)
    for _ in range(SCRATCH_CNT):
        generate_scratch(strokes, MAX_LENGTH, END_BRUSH_RANGE, MID_BRUSH_RANGE)
    exclude_class = 0
    if mask.max()>1 and exclude_random_class:
        exclude_class = np.random.randint(0,mask.max())
    strokes[strokes>0] = mask[strokes>0]
    strokes[strokes==exclude_class] = 0
    return strokes
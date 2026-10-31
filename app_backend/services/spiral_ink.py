"""Colour-agnostic ink detection and measurements for spiral-writing video frames.

Ink is found from contrast, not from one colour range:

* Each pixel is compared with a local paper background. The background is a morphological closing of
  the darkest colour channel, which removes dark strokes narrower than the kernel and keeps slow shading.
  A pixel is a candidate when it is clearly darker than that background. Blue, red, black, grey and most
  other pen inks are dark in their darkest channel, so they all pass.
* Colour is only used to label candidates (blue/violet, red, black/grey, other), not to decide ink.
* Only marks on the paper count. The paper is the light region of the local background, so the table,
  hands and pen bodies are excluded.
* Ruled lines are straight, so long straight runs at any angle are removed on every frame. A camera that
  moves between frames rules out a per-pixel baseline over the video, so none is used.
* Thick objects (pen body, shadows, hand edges) are removed by thickness. Ink strokes are thin.
* A per-pixel hit counter, with a 3x3 tolerance for camera shake, keeps only marks seen in several frames.
  A moving pen or hand leaves few hits, while drawn ink keeps adding them.

All outputs are pixel-level image measurements. Darkness and width are not physical pen pressure.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

COLOUR_NONE, COLOUR_BLUE, COLOUR_RED, COLOUR_NEUTRAL, COLOUR_OTHER = 0, 1, 2, 3, 4
COLOUR_NAMES = {COLOUR_BLUE: "blue_violet", COLOUR_RED: "red", COLOUR_NEUTRAL: "black_or_grey", COLOUR_OTHER: "other_colour"}

INK_DARKNESS_MIN = 40          # levels darker than the local paper background (darkest channel)
PAPER_MIN_LEVEL = 120          # paper is light: the local background must be at least this bright
NEUTRAL_SATURATION_MAX = 35    # below this saturation a dark pixel is black or grey ink (blue pen ink measures about 55)
PERSIST_WINDOW = 10            # recent frames used to judge whether a mark is stable
PERSIST_NEEDED = 6             # frames within that window in which a mark must be a candidate
LINE_LENGTH_FRACTION = 0.20    # straight runs at least this share of the longer frame side are ruling
RULING_GAP = 30                # largest gap in pixels bridged when joining ruling dashes
THICK_FRACTION = 0.02          # half-width (share of the shorter side) above which marks are not strokes
MIN_INK_COMPONENT = 12         # pixels
MIN_INK_PIXELS = 30            # below this the final frame has too little ink to measure
FINAL_WINDOW_FRAMES = 30       # ink seen in the last frames is the drawing; earlier frames add ruling residue
SPIRAL_MIN_RADIUS_FRACTION = 0.08
SPIRAL_MAX_SAMPLES = 3000
SPIRAL_FIT_TOLERANCE = 0.2     # median residual as a share of turn spacing


def _kernel(size: int):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def darkest_channel(frame_bgr) -> np.ndarray:
    return frame_bgr.min(axis=2).astype(np.uint8)


def colour_labels(frame_bgr, candidate) -> np.ndarray:
    """Label candidate pixels by ink family. Unlabelled pixels stay COLOUR_NONE."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    labels = np.zeros(candidate.shape, np.uint8)
    coloured = sat >= NEUTRAL_SATURATION_MAX
    labels[candidate & ~coloured] = COLOUR_NEUTRAL
    blue = coloured & (hue >= 95) & (hue <= 135)
    red = coloured & ((hue <= 10) | (hue >= 165))
    labels[candidate & blue] = COLOUR_BLUE
    labels[candidate & red] = COLOUR_RED
    labels[candidate & coloured & ~blue & ~red] = COLOUR_OTHER
    return labels


def frame_maps(frame_bgr):
    """Darkness of each pixel against its local background, and the paper mask.

    darkness: int16, how much darker a pixel is than the closed (paper-level) background.
    paper: bool, pixels whose surrounding median level is light enough to be paper. Table, hands and pen
    bodies are not paper, so marks there are never ink.
    """
    import cv2
    channel = darkest_channel(frame_bgr)
    short = min(channel.shape)
    size = max(15, (short // 10) | 1)
    background = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, _kernel(size))
    darkness = background.astype(np.int16) - channel.astype(np.int16)
    # A median filter, not the closing, decides paper: closing fills dark table pixels next to the paper edge.
    median = cv2.medianBlur(channel, max(11, (short // 25) | 1))
    paper = median >= PAPER_MIN_LEVEL
    return darkness, paper


def candidate_ink(darkness, paper=None, excluded=None, short_side=None, remove_lines=True):
    """Dark marks on the paper, minus thick objects, straight ruling and excluded areas. Returns a bool mask.

    remove_lines=False is used when the page is aligned to a reference frame. Printed ruling is then removed by the
    reference subtraction, and straight runs in drawn spirals would otherwise be lost as if they were ruling.
    """
    import cv2
    candidate = darkness >= INK_DARKNESS_MIN
    if paper is not None:
        candidate &= cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(9)).astype(bool)
    if excluded is not None:
        candidate &= ~excluded
    short = short_side or min(candidate.shape)

    # Thick objects are not strokes: remove the core and a margin around it.
    binary = candidate.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    thick = (distance > THICK_FRACTION * short).astype(np.uint8)
    if thick.any():
        candidate &= ~(cv2.dilate(thick, _kernel(5)) > 0)

    return remove_ruling(candidate) if remove_lines else candidate


def remove_ruling(mask: np.ndarray) -> np.ndarray:
    """Remove straight runs of marks, which is how printed ruling appears after dashes and gaps.

    Probabilistic Hough lines join dashes that are roughly collinear, with gaps up to GAP pixels, so ruling is
    found even when it is broken into dashes. Only segments at least LINE_LENGTH_FRACTION of the longer frame
    side are removed, which keeps curved strokes and short straight pen strokes.
    """
    import cv2
    binary = mask.astype(np.uint8) * 255
    min_length = max(20, int(max(binary.shape) * LINE_LENGTH_FRACTION))
    segments = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=25, minLineLength=min_length, maxLineGap=RULING_GAP)
    if segments is None:
        return mask.copy()
    ruling = np.zeros_like(binary)
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        cv2.line(ruling, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
    return mask & ~(ruling > 0)


class InkTracker:
    """Temporal support: a mark is ink when it is a candidate in most of the last few frames.

    Each frame's candidates are dilated by one pixel, because a handheld camera moves the strokes slightly.
    A mark needs support in PERSIST_NEEDED of the last PERSIST_WINDOW frames. Drawn ink stays in place, so it
    keeps its support. A pen or hand that passes over a spot, and ruling that flickers as the camera drifts,
    do not reach the threshold.
    """

    def __init__(self, shape, window: int = PERSIST_WINDOW, needed: int = PERSIST_NEEDED):
        self.window = deque(maxlen=window)
        self.total = np.zeros(shape, np.uint8)
        self.needed = needed

    def update(self, candidate) -> np.ndarray:
        import cv2
        tolerant = (cv2.dilate(candidate.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.uint8)
        if len(self.window) == self.window.maxlen:
            self.total -= self.window[0]
        self.window.append(tolerant)
        self.total += tolerant
        return candidate & (self.total >= self.needed)


ALIGN_FEATURES = 2500
ALIGN_MIN_INLIERS = 25         # a frame whose paper matches the reference with fewer inliers is not aligned
ALIGN_RANSAC_PX = 4.0
NEW_INK_MARGIN = 25            # levels darker than the printed mark at the same place in the reference frame
REFERENCE_DILATE = 5           # the printed mark is its darkest value within this many pixels (absorbs a 2 px misfit)
CONFIRM_FRAMES = 3             # ink is accepted once it is new in this many aligned frames


class PaperAligner:
    """Aligns each frame's paper to the reference frame with ORB features and a RANSAC homography.

    The camera and paper move by hand, so a fixed image baseline cannot separate printed content from new ink.
    Only paper features are matched: the hand, skin and table are masked out first.
    """

    def __init__(self, reference_bgr, reference_usable):
        import cv2
        self._cv2 = cv2
        self._orb = cv2.ORB_create(nfeatures=ALIGN_FEATURES, fastThreshold=10)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keypoints, self.descriptors = self._features(reference_bgr, reference_usable)

    def _features(self, frame_bgr, usable):
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Closing removes thin dark lines. Ruling is parallel, so its repeated features slide along the lines and
        # bias the fit. Printed words and paper texture keep their features.
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return self._orb.detectAndCompute(self._clahe.apply(gray), usable.astype(np.uint8) * 255)

    def homography(self, frame_bgr, paper, excluded):
        """Homography mapping reference coordinates to this frame, and its inlier count. None when not aligned."""
        cv2 = self._cv2
        keypoints, descriptors = self._features(frame_bgr, paper & ~excluded)
        if descriptors is None or self.descriptors is None or len(keypoints) < ALIGN_MIN_INLIERS:
            return None, 0
        pairs = self._matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < ALIGN_MIN_INLIERS:
            return None, len(good)
        src = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        dst = np.float32([keypoints[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ALIGN_RANSAC_PX)
        if matrix is None or inliers is None:
            return None, 0
        count = int(inliers.sum())
        # A real page keeps its scale roughly; a degenerate fit can collapse the paper to a line.
        determinant = float(np.linalg.det(matrix[:2, :2]))
        if count < ALIGN_MIN_INLIERS or not 0.3 <= determinant <= 3.0:
            return None, count
        return matrix, count


def reference_marks(reference_darkness):
    """Darkness of the printed marks on the reference frame: pre-existing text, boxes and ruling.

    Each value is the darkest reading within REFERENCE_DILATE pixels, so a small camera error does not turn a
    printed line edge into new ink.
    """
    import cv2
    kernel = np.ones((REFERENCE_DILATE, REFERENCE_DILATE), np.uint8)
    return cv2.dilate(reference_darkness.astype(np.float32), kernel)


def to_reference(candidate, labels, darkness, homography, shape):
    """Move one frame's candidate ink, colour labels and darkness into reference coordinates.

    Persistence is judged in reference coordinates, because a moving camera shifts drawn ink between frames.
    Returns (candidate, labels, darkness), all in reference coordinates.
    """
    import cv2
    height, width = shape
    inverse = np.linalg.inv(homography)
    candidate_ref = cv2.warpPerspective(candidate.astype(np.uint8), inverse, (width, height), flags=cv2.INTER_NEAREST) > 0
    labels_ref = cv2.warpPerspective(labels, inverse, (width, height), flags=cv2.INTER_NEAREST)
    dark_ref = cv2.warpPerspective(darkness.astype(np.float32), inverse, (width, height), flags=cv2.INTER_LINEAR)
    return candidate_ref, labels_ref, dark_ref


def new_ink(stable_ref, dark_ref, marks, reference_paper):
    """Stable ink (reference coordinates) that is not printed on the reference paper.

    Ink is new when it is clearly darker than the printed mark at the same place. A pen stroke that crosses a
    ruling line is darker than the ruling, so it is kept.
    """
    return stable_ref & reference_paper & (dark_ref - marks >= NEW_INK_MARGIN)


class InkAccumulator:
    """Ink in reference coordinates, accepted once it is new in several aligned frames.

    One badly aligned frame can mark a printed edge as new. That mark is not repeated in other frames, so it
    never reaches the accumulated ink. Drawn ink is new in every frame after it is drawn, so it is accepted.
    """

    def __init__(self, shape, confirm: int = CONFIRM_FRAMES):
        self.hits = np.zeros(shape, np.uint16)
        self.confirm = confirm
        self.ever = np.zeros(shape, bool)

    def update(self, fresh) -> np.ndarray:
        """Add one frame's new ink. Returns the pixels that became accepted in this frame."""
        self.hits += fresh.astype(np.uint16)
        before = self.ever
        self.ever = self.hits >= self.confirm
        return self.ever & ~before


def project_mask(mask_ref, homography, frame_shape):
    """Reference-coordinate mask moved into this frame, for drawing on the video."""
    import cv2
    height, width = frame_shape[:2]
    return cv2.warpPerspective(mask_ref.astype(np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST) > 0


def zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Morphological thinning (Zhang-Suen), vectorised. Returns a one-pixel-wide skeleton."""
    img = np.pad(binary.astype(np.uint8), 1)
    while True:
        removed = 0
        for step in (0, 1):
            p2 = img[:-2, 1:-1].astype(np.int16)
            p3 = img[:-2, 2:].astype(np.int16)
            p4 = img[1:-1, 2:].astype(np.int16)
            p5 = img[2:, 2:].astype(np.int16)
            p6 = img[2:, 1:-1].astype(np.int16)
            p7 = img[2:, :-2].astype(np.int16)
            p8 = img[1:-1, :-2].astype(np.int16)
            p9 = img[:-2, :-2].astype(np.int16)
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            centre = img[1:-1, 1:-1] == 1
            mark = centre & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & c1 & c2
            inner = img[1:-1, 1:-1]
            inner[mark] = 0
            removed += int(mark.sum())
        if removed == 0:
            break
    return img[1:-1, 1:-1].astype(bool)


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad(skeleton.astype(np.uint8), 1)
    total = np.zeros(skeleton.shape, np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            total += padded[1 + dy:1 + dy + skeleton.shape[0], 1 + dx:1 + dx + skeleton.shape[1]]
    return total * skeleton


def skeleton_length(skeleton: np.ndarray) -> float:
    """Approximate path length: horizontal and vertical links count 1, diagonal links count sqrt(2)."""
    s = skeleton.astype(np.uint8)
    horizontal = int(np.sum(s[:, 1:] & s[:, :-1]) + np.sum(s[1:, :] & s[:-1, :]))
    diagonal = int(np.sum(s[1:, 1:] & s[:-1, :-1]) + np.sum(s[1:, :-1] & s[:-1, 1:]))
    return float(horizontal + math.sqrt(2) * diagonal)


def ink_measurements(stable: np.ndarray, labels: np.ndarray, darkness: np.ndarray, short_side: int, frame_area: float):
    """Ink and stroke values from one frame. Returns (measured, estimated, unavailable_reason or None)."""
    import cv2
    pixels = int(stable.sum())
    if pixels < MIN_INK_PIXELS:
        return {}, {}, "Too little ink was detected on the final frame to measure strokes."

    measured: dict = {}
    estimated: dict = {}
    ys, xs = np.nonzero(stable)
    bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    measured["ink_coverage_of_frame"] = round(pixels / frame_area, 5)
    darkness_values = darkness[stable].astype(float)
    measured["ink_darkness_mean"] = round(float(darkness_values.mean()), 1)
    measured["ink_darkness_median"] = round(float(np.median(darkness_values)), 1)
    colour_values = labels[stable]
    total = float(colour_values.size)
    measured["ink_colour_share"] = {COLOUR_NAMES[c]: round(float(np.sum(colour_values == c)) / total, 3)
                                    for c in COLOUR_NAMES}
    measured["ink_density_in_bounding_box"] = round(pixels / bbox_area, 4)

    binary = stable.astype(np.uint8)
    count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= MIN_INK_COMPONENT) + 1
    components = int(keep.size)
    measured["ink_components"] = components
    estimated["gaps_between_pieces"] = max(components - 1, 0)
    largest_share = float(areas.max() / areas.sum()) if areas.size else 0.0
    estimated["continuity_largest_piece_share"] = round(largest_share, 3)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    skeleton = zhang_suen(stable)
    widths = distance[skeleton] * 2.0 if skeleton.any() else distance[stable] * 2.0
    if widths.size:
        measured["stroke_width_px_mean"] = round(float(widths.mean()), 2)
        measured["stroke_width_px_median"] = round(float(np.median(widths)), 2)
        measured["stroke_width_px_std"] = round(float(widths.std()), 2)
        estimated["stroke_width_variation"] = round(float(widths.std() / widths.mean()), 3) if widths.mean() > 0 else None
    measured["stroke_path_length_px"] = round(skeleton_length(skeleton), 1)
    neighbours = _neighbour_count(skeleton)
    endpoints = int(np.sum(neighbours == 1))
    estimated["stroke_endpoints"] = endpoints
    estimated["path_length_share_of_frame_side"] = round(skeleton_length(skeleton) / max(short_side, 1), 3)
    return measured, estimated, None


MIN_FRONT_PIXELS = 8           # a frame must add this much new ink to count as drawing
PAUSE_SECONDS = 0.5            # a gap of at least this long between drawn frames is a pause
MIN_FRONT_EVENTS = 5


def drawing_timeline(events, fps: float, frame_shape):
    """Timing and path of the newly drawn ink. Returns (estimated values, unavailable reason or None).

    events: (frame index, x, y) for frames that added ink, in reference coordinates. The point is the centroid of
    the new ink in that frame. It is a proxy for the drawing point, not a tracked pen tip, and the persistence
    check adds a delay of a few frames to every event.
    """
    if len(events) < MIN_FRONT_EVENTS or fps <= 0:
        return {}, "Too little newly drawn ink to estimate the drawing timing."
    estimated: dict = {}
    frames = np.array([e[0] for e in events], dtype=float)
    points = np.array([[e[1], e[2]] for e in events], dtype=float)
    estimated["drawing_duration_seconds"] = round(float((frames[-1] - frames[0]) / fps), 2)
    gaps = np.diff(frames) / fps
    estimated["pauses"] = int(np.sum(gaps >= PAUSE_SECONDS))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    side = float(max(frame_shape[:2]))
    estimated["front_path_length_frame_sides"] = round(float(steps.sum()) / side, 3)
    moving = gaps > 0
    if moving.any():
        speeds = steps[moving] / gaps[moving] / side
        estimated["front_speed_median_frame_sides_per_s"] = round(float(np.median(speeds)), 3)
    return estimated, None


def isolate_spiral(ink_mask: np.ndarray):
    """Pick the spiral out of all ink on the page, so other handwriting is not measured as the spiral.

    Nearby ink is grouped, and each of the largest groups is tested with the spiral fit. The largest group that
    fits is the spiral. Returns (spiral mask or None, fit dict or reason string, pixels ignored).
    """
    import cv2
    total = int(ink_mask.sum())
    if total < 80:
        return None, "Too little ink to isolate a spiral.", total
    grouped = cv2.dilate(ink_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    best = None
    last_reason = "No group of ink on the page forms a spiral."
    for group in order[:6]:
        mask = (labels == group) & ink_mask
        if int(mask.sum()) < 80:
            continue
        fit = spiral_fit(mask)
        if isinstance(fit, str):
            last_reason = fit
            continue
        if best is None or int(mask.sum()) > int(best[0].sum()):
            best = (mask, fit)
    if best is None:
        return None, last_reason, total
    return best[0], best[1], total - int(best[0].sum())


def spiral_fit(stable: np.ndarray):
    """Fit an Archimedean spiral r = c + (pitch / 2pi) * theta (mod pitch) to the ink.

    The centre is searched around the ink centroid, because dense outer loops pull the centroid away from
    the true centre. Returns a dict of geometry values, or a reason string when the ink does not form a spiral.
    The fit is in image pixels, without perspective correction.
    """
    ys, xs = np.nonzero(stable)
    if xs.size < 80:
        return "Too little ink to fit a spiral."
    skeleton = zhang_suen(stable)
    sy, sx = np.nonzero(skeleton)
    if sx.size > SPIRAL_MAX_SAMPLES:
        step = int(math.ceil(sx.size / SPIRAL_MAX_SAMPLES))
        sx, sy = sx[::step], sy[::step]
    sx, sy = sx.astype(float), sy.astype(float)
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    rough_r = float(np.hypot(sx - cx0, sy - cy0).max())
    if rough_r < SPIRAL_MIN_RADIUS_FRACTION * min(stable.shape):
        return "The ink is too small to fit a spiral."

    # Search centres within 15% of the rough radius of the centroid, on a 5 x 5 grid.
    spread = 0.15 * rough_r
    candidates = []
    for cx in np.linspace(cx0 - spread, cx0 + spread, 5):
        for cy in np.linspace(cy0 - spread, cy0 + spread, 5):
            radius = np.hypot(sx - cx, sy - cy)
            theta = np.mod(np.arctan2(sy - cy, sx - cx), 2 * math.pi)
            r_max = float(radius.max())
            # Pitch between r_max/10 and r_max/2.5 means 2.5 to 10 turns. Finer pitches fit any dense handwriting.
            for pitch in np.linspace(max(3.0, r_max / 10), r_max / 2.5, 30):
                for offset in np.linspace(0.0, pitch, 24, endpoint=False):
                    residual = np.mod(radius - offset - pitch * theta / (2 * math.pi) + pitch / 2, pitch) - pitch / 2
                    candidates.append((float(np.median(np.abs(residual))), float(pitch), float(offset), float(cx), float(cy)))
    # A spiral with pitch p also lies on a lattice of pitch p/2, so the smallest pitch fits too. The true pitch
    # is the largest one that still fits nearly as well as the best.
    best_score = min(c[0] for c in candidates)
    score, pitch, offset, cx, cy = max((c for c in candidates if c[0] <= best_score * 1.15 + 0.5), key=lambda c: c[1])

    radius = np.hypot(sx - cx, sy - cy)
    theta = np.mod(np.arctan2(sy - cy, sx - cx), 2 * math.pi)
    r_max = float(radius.max())
    residual = np.mod(radius - offset - pitch * theta / (2 * math.pi) + pitch / 2, pitch) - pitch / 2
    turns = r_max / pitch
    if turns < 1.5:
        return "The ink does not reach enough turns to form a spiral."
    if score > SPIRAL_FIT_TOLERANCE * pitch:
        return "The ink does not follow a spiral pattern around one centre."

    angles = np.floor(theta / (2 * math.pi) * 36).astype(int) % 36
    coverage = len(set(angles.tolist())) / 36.0
    return {
        "centre_px": [round(cx, 1), round(cy, 1)],
        "turn_spacing_px": round(float(pitch), 2),
        "outer_radius_px": round(r_max, 1),
        "turns_detected": round(float(turns), 2),
        "fit_rms_deviation_px": round(float(np.sqrt(np.mean(residual ** 2))), 2),
        "fit_deviation_share_of_spacing": round(float(np.median(np.abs(residual))) / pitch, 3),
        "radius_variation_share_of_spacing": round(float(residual.std()) / pitch, 3),
        "angular_coverage": round(coverage, 3),
    }

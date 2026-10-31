"""Hand-landmark worker. Runs in a separate Python environment that has MediaPipe installed.

Usage: python -I hand_landmark_worker.py <video path> <output json path>

The main server runs this file as a subprocess, so MediaPipe and its NumPy version never load into the
server process. The hand model ships inside the MediaPipe wheel (hand_landmark_full.tflite), so no
separate weights are downloaded. The output holds one entry per decoded frame: null when no hand passed
MediaPipe's detection threshold, otherwise the handedness, its score and the 21 normalised landmarks.
"""
import json
import sys


def main(video_path: str, output_path: str) -> None:
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit("video could not be opened")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        # Measured on the 16.7 s spiral recording (393 frames): 0.5 found 62 frames, 0.3 found 201 and 0.15 found
        # 337. At 0.3 the hand was missed in frames where it was plainly visible; at 0.15 the landmarks followed
        # the fingers in every frame checked. Weaker detections are still filtered by the handedness score.
        min_detection_confidence=0.15,
        min_tracking_confidence=0.15,
    )
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            entry = None
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0].landmark
                handedness = result.multi_handedness[0].classification[0]
                entry = {
                    "handedness": handedness.label,
                    "score": round(float(handedness.score), 4),
                    "landmarks": [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in landmarks],
                }
            frames.append(entry)
    finally:
        hands.close()
        capture.release()
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"width": width, "height": height, "frames": frames}, handle)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: hand_landmark_worker.py <video> <output.json>")
    main(sys.argv[1], sys.argv[2])

"""
===================================================================================================================
Notes
===================================================================================================================
- facing direction = 1 for cam 0 (human's right side); -1 for cam 1
- extension = -angle; flexion = +angle

===================================================================================================================
KIV
===================================================================================================================

## A: Neck, Trunk & Leg Analysis
- additional score is added to ALL frames regardless

## B: Arm & Wrist Analysis
- arm supported/leaning score is added to ALL frames regardless
- additional score is added to ALL frames regardless

## Misc
- delay in live view (about 1s) -> resolution down sampling does not help

===================================================================================================================
To Be Solved
===================================================================================================================


"""

import cv2
import threading
import queue
import time
from datetime import datetime
import numpy as np
import csv
import subprocess
import os
import shutil
import mediapipe as mp
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
import itertools
from PIL import Image, ImageTk
import copy


class DualCameraCapture:
    PAIRED_LANDMARKS = {
        2: 5,  # left_eye ↔ right_eye
        7: 8,  # left_ear ↔ right_ear
        11: 12,  # left_shoulder ↔ right_shoulder
        13: 14,  # left_elbow ↔ right_elbow
        15: 16,  # left_wrist ↔ right_wrist
        17: 18,  # left_pinky ↔ right_pinky
        19: 20,  # left_index ↔ right_index
        21: 22,  # left_thumb ↔ right_thumb
        23: 24,  # left_hip ↔ right_hip
        25: 26,  # left_knee ↔ right_knee
        27: 28,  # left_ankle ↔ right_ankle
        29: 30,  # left_heel ↔ right_heel
        31: 32,  # left_foot_index ↔ right_foot_index
    }

    def __init__(self, cam0_idx=0, cam1_idx=1, fps=30):
        self.cam0_idx = cam0_idx
        self.cam1_idx = cam1_idx
        self.fps = fps

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create pose detectors for both cameras
        self.pose0 = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose1 = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe landmark names (33 landmarks)
        self.landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        # EMA state for both cameras
        self.ema_landmarks0 = (
            None  # Store the smoothed positions for cam 0 (start with 0)
        )
        self.ema_landmarks1 = None
        self.ema_alpha = 0.7

        # Stability tracking with 10-frame rolling window
        self.stability_window_size = 10
        self.stability_history0 = [
            deque(maxlen=self.stability_window_size) for _ in range(33)
        ]  # deque of size 10 for every landmark (10x33)
        self.stability_history1 = [
            deque(maxlen=self.stability_window_size) for _ in range(33)
        ]
        self.prev_landmarks0 = None  # Previous frame landmarks for distance calculation
        self.prev_landmarks1 = None

        # Initialize cameras
        self.cap0 = cv2.VideoCapture(cam0_idx)
        self.cap1 = cv2.VideoCapture(cam1_idx)

        # Check if cameras opened successfully
        if not self.cap0.isOpened():
            raise Exception(f"Cannot open camera {cam0_idx}")
        if not self.cap1.isOpened():
            raise Exception(f"Cannot open camera {cam1_idx}")

        # Set FPS
        self.cap0.set(cv2.CAP_PROP_FPS, fps)
        self.cap1.set(cv2.CAP_PROP_FPS, fps)

        # Get native resolutions
        self.width0 = int(self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height0 = int(self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # After 90° rotation, width and height swap
        self.rotated_width0 = self.height0
        self.rotated_height0 = self.width0
        self.rotated_width1 = self.height1
        self.rotated_height1 = self.width1

        # Get actual FPS from cameras
        self.actual_fps0 = self.cap0.get(cv2.CAP_PROP_FPS)
        self.actual_fps1 = self.cap1.get(cv2.CAP_PROP_FPS)

        print(
            f"Camera 0 resolution: {self.width0}x{self.height0} -> After rotation: {self.rotated_width0}x{self.rotated_height0}"
        )
        print(f"Camera 0 reported FPS: {self.actual_fps0}")
        print(
            f"Camera 1 resolution: {self.width1}x{self.height1} -> After rotation: {self.rotated_width1}x{self.rotated_height1}"
        )
        print(f"Camera 1 reported FPS: {self.actual_fps1}")

        # Measured FPS from calibration
        self.measured_fps0 = None
        self.measured_fps1 = None
        self.synchronized_fps = None

        # Queues for synchronized frame capture
        self.queue0 = queue.Queue(maxsize=10)
        self.queue1 = queue.Queue(maxsize=10)

        # Queues for MediaPipe processed results
        self.processed_queue0 = queue.Queue(maxsize=10)
        self.processed_queue1 = queue.Queue(maxsize=10)

        # Last good frames for duplication
        self.last_frame0 = None
        self.last_frame1 = None

        # Frame statistics
        self.duplicated_frames0 = 0
        self.duplicated_frames1 = 0
        self.dropped_frames0 = 0
        self.dropped_frames1 = 0
        self.total_frames_written = 0

        # Control flags
        self.running = False
        self.recording = False

        # Video writers
        self.out0 = None
        self.out1 = None

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename0 = f"camera0_{timestamp}.mp4"
        self.filename1 = f"camera1_{timestamp}.mp4"
        self.temp_filename0 = f"camera0_{timestamp}_temp.mp4"
        self.temp_filename1 = f"camera1_{timestamp}_temp.mp4"
        self.csv_filename0 = f"camera_{timestamp}_cam0_frames.csv"
        self.csv_filename1 = f"camera_{timestamp}_cam1_frames.csv"

        # Frame data for CSV logging
        self.frame_data0 = []
        self.frame_data1 = []

        # Buffers for matching frames by frame number
        self.buffer0 = {}
        self.buffer1 = {}

        # Raw timestamps for true FPS calculation
        self.frame_timestamps0 = []
        self.frame_timestamps1 = []

        # Timing data
        self.true_fps = None

        # neck twist & side bend
        self.neck_twisted = None
        self.neck_side_bent = None

        # trunk "upright" reference
        self.trunk_baseline_angle0 = None
        self.trunk_baseline_angle1 = None

        # trunk twist & side bend
        self.trunk_twisted = None
        self.trunk_side_bent = None

        # leg raised baseline
        self.leg_raised_baseline_mean0 = None
        self.leg_raised_baseline_mean1 = None

        # natural angle baseline
        self.neck_baseline_angle0 = None
        self.left_knee_baseline_angle0 = None
        self.right_knee_baseline_angle0 = None
        self.left_upper_arm_baseline_angle0 = None
        self.right_upper_arm_baseline_angle0 = None
        self.left_lower_arm_baseline_angle0 = None
        self.right_lower_arm_baseline_angle0 = None
        self.left_wrist_baseline_angle0 = None
        self.right_wrist_baseline_angle0 = None

        # Force/Load Score
        self.force_load_score = None

        # upper arm baseline
        self.shoulder_raised = None
        self.upper_arm_abducted = None
        self.arm_supported_score = None

        # wrist baseline
        self.left_wrist_bent = None
        self.right_wrist_bent = None
        self.left_wrist_twisted = None
        self.right_wrist_twisted = None

        # Coupling score
        self.coupling_score = None

        # Activity score
        self.activity_score = None

        # Camera Fusion
        self.z_baseline0 = None  # list of 33 mean Z values from cam0
        self.z_baseline1 = None  # list of 33 mean Z values from cam1
        self.last_valid_trunk_length0 = None
        self.last_valid_trunk_length1 = None
        self.trunk_length_baseline_mean0 = None
        self.trunk_length_baseline_mean1 = None
        self.cam1_projection_x_offsets = (
            {}
        )  # per-landmark x offsets, keyed by landmark index

        # Storage for limb length calibration (Left Joint 1, Left Joint 2) mapping
        self.calib_limb_lengths = {
            (11, 13): [],
            (13, 15): [],  # Arm
            (15, 19): [],
            (15, 17): [],
            (15, 21): [],  # Hand
            (23, 25): [],
            (25, 27): [],  # Leg
            (27, 31): [],
            (27, 29): [],  # Foot
        }
        self.max_limb_ratios = {}  # Final computed ratios

        """ uncomment this to turn off the weighted fusion tuning
        # Initialize Weight Tuning CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tuning_csv_path = f'weight_tuning_results_{timestamp}.csv'
        weights = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]
        headers = []
        for w in weights:
            headers.extend(
                [f'trunk_angle_{w}', f'neck_angle_{w}', f'r_upper_arm_{w}', f'r_lower_arm_{w}', f'leg_max_angle_{w}'])

        with open(self.tuning_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        """  # uncomment this to turn off the weighted fusion tuning

    def rotate_frame_90cw(self, frame):
        """Rotate frame 90 degrees clockwise"""
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    def apply_ema(self, current_landmarks, previous_ema):
        """Apply EMA smoothing to landmark coordinates"""
        if previous_ema is None:
            return current_landmarks.copy()

        smoothed = []
        for curr, prev in zip(current_landmarks, previous_ema):
            smoothed_landmark = {
                "x": self.ema_alpha * curr["x"] + (1 - self.ema_alpha) * prev["x"],
                "y": self.ema_alpha * curr["y"] + (1 - self.ema_alpha) * prev["y"],
                "z": self.ema_alpha * curr["z"] + (1 - self.ema_alpha) * prev["z"],
                "visibility": curr["visibility"],
            }
            smoothed.append(smoothed_landmark)
        return smoothed

    def calculate_stability(
        self, current_landmarks, previous_landmarks, stability_history
    ):
        """Calculate stability for each landmark using 10-frame rolling average"""
        stability_scores = []

        if previous_landmarks is None:
            # noinspection PyTypeChecker
            return [float("nan")] * 33

        for i, (curr, prev) in enumerate(zip(current_landmarks, previous_landmarks)):
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            dz = curr["z"] - prev["z"]
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)

            stability_history[i].append(distance)

            if len(stability_history[i]) > 0:
                avg_distance = np.mean(stability_history[i])
                stability_scores.append(avg_distance)
            else:
                stability_scores.append(float("nan"))

        return stability_scores

    def reset_stability(self, stability_history):
        """Reset stability history when pose is lost"""
        for i in range(33):
            stability_history[i].clear()

    ################################################# REBA SECTION A #######################################################
    # NECK ---------------------------------------------------------------------------------------------------------------
    ## Neck Angle
    def calculate_neck_angle(
        self, landmarks, facing_direction, frame_width, frame_height
    ):
        """
        Calculate neck flexion/extension angle relative to trunk.
        facing_direction: +1 if person's front is in positive x direction (cam0),
                          -1 if negative x direction (cam1)
        Returns angle in degrees (positive = flexion, negative = extension)
        """

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])
        left_ear = scale(landmarks[7])
        right_ear = scale(landmarks[8])

        """
        # Debug
        print(f"[Neck] L Shoulder (norm): ({landmarks[11]['x']:.4f}, {landmarks[11]['y']:.4f})")
        print(f"[Neck] R Shoulder (norm): ({landmarks[12]['x']:.4f}, {landmarks[12]['y']:.4f})")
        print(f"[Neck] L Hip (norm):      ({landmarks[23]['x']:.4f}, {landmarks[23]['y']:.4f})")
        print(f"[Neck] R Hip (norm):      ({landmarks[24]['x']:.4f}, {landmarks[24]['y']:.4f})")
        print(f"[Neck] L Ear (norm):      ({landmarks[7]['x']:.4f}, {landmarks[7]['y']:.4f})")
        print(f"[Neck] R Ear (norm):      ({landmarks[8]['x']:.4f}, {landmarks[8]['y']:.4f})")
        print(f"[Neck] L Shoulder: ({left_shoulder['x']:.1f}, {left_shoulder['y']:.1f})")
        print(f"[Neck] R Shoulder: ({right_shoulder['x']:.1f}, {right_shoulder['y']:.1f})")
        print(f"[Neck] L Hip:      ({left_hip['x']:.1f}, {left_hip['y']:.1f})")
        print(f"[Neck] R Hip:      ({right_hip['x']:.1f}, {right_hip['y']:.1f})")
        print(f"[Neck] L Ear:      ({left_ear['x']:.1f}, {left_ear['y']:.1f})")
        print(f"[Neck] R Ear:      ({right_ear['x']:.1f}, {right_ear['y']:.1f})")
        """

        shoulder_mid = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) / 2,
                (left_shoulder["y"] + right_shoulder["y"]) / 2,
            ]
        )

        hip_mid = np.array(
            [(left_hip["x"] + right_hip["x"]) / 2, (left_hip["y"] + right_hip["y"]) / 2]
        )

        ear_mid = np.array(
            [(left_ear["x"] + right_ear["x"]) / 2, (left_ear["y"] + right_ear["y"]) / 2]
        )

        trunk_vec = shoulder_mid - hip_mid
        neck_vec = ear_mid - shoulder_mid

        cos_angle = np.dot(trunk_vec, neck_vec) / (
            np.linalg.norm(trunk_vec) * np.linalg.norm(neck_vec) + 1e-8
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        cross_z = trunk_vec[0] * neck_vec[1] - trunk_vec[1] * neck_vec[0]
        if facing_direction * cross_z < 0:
            angle = -angle

        """
        # Debug
        print(f"[Neck] Shoulder Mid: ({shoulder_mid[0]:.1f}, {shoulder_mid[1]:.1f})")
        print(f"[Neck] Hip Mid:      ({hip_mid[0]:.1f}, {hip_mid[1]:.1f})")
        print(f"[Neck] Ear Mid:      ({ear_mid[0]:.1f}, {ear_mid[1]:.1f})")
        print(f"[Neck] Trunk Vec: ({trunk_vec[0]:.1f}, {trunk_vec[1]:.1f})")
        print(f"[Neck] Neck Vec:  ({neck_vec[0]:.1f}, {neck_vec[1]:.1f})")
        print(f"[Neck] Angle: {angle:.2f} deg | cross_z: {cross_z:.4f} | facing: {facing_direction}")
        print("-" * 40)
        """

        # Cap Neck flexion: -50 to 80
        angle = float(np.clip(angle, -50, 80))

        return angle

    def calculate_neck_reba_score(self, angle):
        """
        REBA neck score based on angle.
        Returns 1 or 2, or NaN if angle is NaN.
        """
        if np.isnan(angle):
            return float("nan")
        if angle < 0 or angle > 20:
            return 2
        return 1

    ## Neck Twist & Side Bend
    def get_neck_adjustments(self):
        """Prompt user for neck twist and side bend"""
        while True:
            twisted = input("\nWas the neck twisted? (y/n): ").strip().lower()
            if twisted in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            side_bent = input("Was the neck side bent? (y/n): ").strip().lower()
            if side_bent in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        return twisted == "y", side_bent == "y"

    ## Neck Side Bend
    def calculate_nose_shoulder_ratio(self, landmarks, frame_width, frame_height):
        """Side bending metric = (vertical distance between left ear and shoulder midpoint) / (vertical distance between hip midpoint and shoulder midpoint)"""

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_ear = scale(landmarks[7])
        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])

        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        hip_mid_y = (left_hip["y"] + right_hip["y"]) / 2

        trunk_length = abs(hip_mid_y - shoulder_mid_y) + 1e-8
        ear_shoulder_y_dist = abs(left_ear["y"] - shoulder_mid_y)

        return ear_shoulder_y_dist / trunk_length

    def calculate_neck_side_bend_angle(self, landmarks, frame_width, frame_height):
        """Neck side bend angle = angle between neck vector (shoulder mid -> ear mid) and trunk vector (hip mid -> shoulder mid)"""

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_ear = scale(landmarks[7])
        right_ear = scale(landmarks[8])
        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])

        ear_mid = np.array(
            [(left_ear["x"] + right_ear["x"]) / 2, (left_ear["y"] + right_ear["y"]) / 2]
        )
        shoulder_mid = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) / 2,
                (left_shoulder["y"] + right_shoulder["y"]) / 2,
            ]
        )
        hip_mid = np.array(
            [(left_hip["x"] + right_hip["x"]) / 2, (left_hip["y"] + right_hip["y"]) / 2]
        )

        neck_vec = ear_mid - shoulder_mid
        trunk_vec = shoulder_mid - hip_mid

        cos_angle = np.dot(neck_vec, trunk_vec) / (
            np.linalg.norm(neck_vec) * np.linalg.norm(trunk_vec) + 1e-8
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def calculate_normalized_shoulder_width(self, landmarks, frame_width, frame_height):
        """Normalized shoulder width = shoulder X distance / 3D trunk length"""

        def scale(lm):
            return {
                "x": lm["x"] * frame_width,
                "y": lm["y"] * frame_height,
                "z": lm["z"] * frame_width,
            }

        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])

        shoulder_mid = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) / 2,
                (left_shoulder["y"] + right_shoulder["y"]) / 2,
                (left_shoulder["z"] + right_shoulder["z"]) / 2,
            ]
        )
        hip_mid = np.array(
            [
                (left_hip["x"] + right_hip["x"]) / 2,
                (left_hip["y"] + right_hip["y"]) / 2,
                (left_hip["z"] + right_hip["z"]) / 2,
            ]
        )

        trunk_length_3d = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8
        shoulder_x_dist = abs(left_shoulder["x"] - right_shoulder["x"])
        normalized = shoulder_x_dist / trunk_length_3d
        return max(normalized, 0.05)

    def calculate_shoulder_width_correction(
        self, landmarks, frame_width, frame_height, baseline
    ):
        """Correction factor = current normalized shoulder width / baseline"""
        if baseline is None:
            return 1.0
        current = self.calculate_normalized_shoulder_width(
            landmarks, frame_width, frame_height
        )
        return current / baseline

    def calculate_neck_side_bend_roll(self, landmarks):
        """
        Compute head roll relative to body coordinate system.
        Uses raw normalized landmarks (no scaling needed).
        Returns roll component = projection of head plane normal onto body X axis.
        """
        # Body landmarks
        left_shoulder = np.array(
            [landmarks[11]["x"], landmarks[11]["y"], landmarks[11]["z"]]
        )
        right_shoulder = np.array(
            [landmarks[12]["x"], landmarks[12]["y"], landmarks[12]["z"]]
        )
        left_hip = np.array(
            [landmarks[23]["x"], landmarks[23]["y"], landmarks[23]["z"]]
        )
        right_hip = np.array(
            [landmarks[24]["x"], landmarks[24]["y"], landmarks[24]["z"]]
        )

        # Head landmarks
        left_eye = np.array([landmarks[2]["x"], landmarks[2]["y"], landmarks[2]["z"]])
        right_eye = np.array([landmarks[5]["x"], landmarks[5]["y"], landmarks[5]["z"]])
        nose = np.array([landmarks[0]["x"], landmarks[0]["y"], landmarks[0]["z"]])

        # Body coordinate system
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2

        # Y-axis using only X and Y coords for stability
        shoulder_mid_2d = np.array([shoulder_mid[0], shoulder_mid[1], 0])
        hip_mid_2d = np.array([hip_mid[0], hip_mid[1], 0])
        y_axis = shoulder_mid_2d - hip_mid_2d
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        shoulder_vec = left_shoulder - right_shoulder
        shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)

        z_axis = np.cross(y_axis, shoulder_vec)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Head plane normal
        eye_vec = right_eye - left_eye
        v2 = nose - left_eye
        head_normal = np.cross(eye_vec, v2)
        head_normal = head_normal / (np.linalg.norm(head_normal) + 1e-8)

        # Head up vector (toward scalp)
        head_up = np.cross(head_normal, eye_vec)
        head_up = head_up / (np.linalg.norm(head_up) + 1e-8)

        # Ensure head_up points toward scalp (aligns with body Y axis)
        if np.dot(head_up, y_axis) < 0:
            head_up = -head_up

        # Roll = projection of head up onto body X axis
        roll = np.dot(head_up, x_axis)

        # Debug
        print(f"[Roll Debug] y_axis: {y_axis}")
        print(f"[Roll Debug] shoulder_vec: {shoulder_vec}")
        print(f"[Roll Debug] z_axis: {z_axis}")
        print(f"[Roll Debug] x_axis: {x_axis}")
        print(f"[Roll Debug] head_normal: {head_normal}")
        print(f"[Roll Debug] head_up: {head_up}")
        print(f"[Roll Debug] dot(head_up, y_axis): {np.dot(head_up, y_axis):.4f}")

        return abs(roll)

    def interpolate_roll_baseline(self, current_shoulder_width, calib_points):
        """
        Interpolate roll baseline based on current shoulder width.
        calib_points: list of (shoulder_width, roll_baseline) sorted by shoulder_width.
        Clamps if outside calibrated range.
        """
        if not calib_points:
            return 0.0

        # Sort by shoulder width
        sorted_points = sorted(calib_points, key=lambda p: p[0])

        # Clamp if outside range
        if current_shoulder_width <= sorted_points[0][0]:
            return sorted_points[0][1]
        if current_shoulder_width >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Find two nearest points and interpolate
        for i in range(len(sorted_points) - 1):
            sw0, rb0 = sorted_points[i]
            sw1, rb1 = sorted_points[i + 1]
            if sw0 <= current_shoulder_width <= sw1:
                t = (current_shoulder_width - sw0) / (sw1 - sw0 + 1e-8)
                return rb0 + t * (rb1 - rb0)

        return sorted_points[-1][1]

    def calculate_neck_side_bend(self, landmarks, roll_baseline):
        """Returns True if neck side bending detected using body-relative head roll"""
        if roll_baseline is None:
            return False
        roll = self.calculate_neck_side_bend_roll(landmarks)

        # Debug
        print(f"[Neck Side Bend] roll_baseline_mean: {roll_baseline:.4f}")
        print(f"[Neck Side Bend] roll: {roll:.4f}")
        print(f"[Neck Side Bend] abs deviation: {abs(roll - roll_baseline):.4f}")
        print(f"[Neck Side Bend] threshold: 0.2")
        print("-" * 40)

        return bool(abs(roll - roll_baseline) > 0.2)

    # TRUNK ---------------------------------------------------------------------------------------------------------------
    ## Trunk Angle
    def calculate_trunk_angle(
        self, landmarks, facing_direction, frame_width, frame_height
    ):
        """Calculate trunk flexion/extension angle relative to upright baseline"""

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])

        shoulder_mid = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) / 2,
                (left_shoulder["y"] + right_shoulder["y"]) / 2,
            ]
        )
        hip_mid = np.array(
            [(left_hip["x"] + right_hip["x"]) / 2, (left_hip["y"] + right_hip["y"]) / 2]
        )

        trunk_vec = shoulder_mid - hip_mid
        vertical_vec = np.array([0, -1])  # upward in image coordinates

        cos_angle = np.dot(trunk_vec, vertical_vec) / (
            np.linalg.norm(trunk_vec) * np.linalg.norm(vertical_vec) + 1e-8
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # Determine flexion vs extension using cross product
        cross_z = trunk_vec[1] * vertical_vec[0] - trunk_vec[0] * vertical_vec[1]
        if facing_direction * cross_z < 0:
            angle = -angle  # extension

        """
        # Debug
        # Normalised coordinates
        print(f"[Trunk] L Shoulder (norm): ({landmarks[11]['x']:.4f}, {landmarks[11]['y']:.4f})")
        print(f"[Trunk] R Shoulder (norm): ({landmarks[12]['x']:.4f}, {landmarks[12]['y']:.4f})")
        print(f"[Trunk] L Hip (norm):      ({landmarks[23]['x']:.4f}, {landmarks[23]['y']:.4f})")
        print(f"[Trunk] R Hip (norm):      ({landmarks[24]['x']:.4f}, {landmarks[24]['y']:.4f})")

        # Scaled coordinates
        print(f"[Trunk] L Shoulder: ({left_shoulder['x']:.1f}, {left_shoulder['y']:.1f})")
        print(f"[Trunk] R Shoulder: ({right_shoulder['x']:.1f}, {right_shoulder['y']:.1f})")
        print(f"[Trunk] L Hip:      ({left_hip['x']:.1f}, {left_hip['y']:.1f})")
        print(f"[Trunk] R Hip:      ({right_hip['x']:.1f}, {right_hip['y']:.1f})")

        # Midpoints and vectors
        print(f"[Trunk] Shoulder Mid: ({shoulder_mid[0]:.1f}, {shoulder_mid[1]:.1f})")
        print(f"[Trunk] Hip Mid:      ({hip_mid[0]:.1f}, {hip_mid[1]:.1f})")
        print(f"[Trunk] Trunk Vec:    ({trunk_vec[0]:.1f}, {trunk_vec[1]:.1f})")
        print(f"[Trunk] Vertical Vec: ({vertical_vec[0]:.1f}, {vertical_vec[1]:.1f})")
        print(f"[Trunk] cross_z: {cross_z:.4f} | facing: {facing_direction}")
        print(f"[Trunk] Angle: {angle:.1f} deg")
        print("-" * 40)
        """

        angle = float(np.clip(angle, -40, 70))

        return angle

    def calculate_trunk_reba_score(self, angle, baseline_angle):
        """REBA trunk score based on angle relative to upright baseline"""
        if np.isnan(angle) or np.isnan(baseline_angle):
            return float("nan")

        relative_angle = angle - baseline_angle

        if abs(relative_angle) < 0.01:
            return 1
        elif relative_angle > 0:  # flexion
            if relative_angle <= 20:
                return 2
            elif relative_angle <= 60:
                return 3
            else:
                return 4
        else:  # extension
            abs_angle = abs(relative_angle)
            if abs_angle <= 20:
                return 2
            else:
                return 3

    ## Trunk Twist & Side Bend
    def get_trunk_adjustments(self):
        """Prompt user for trunk twist and side bend"""
        while True:
            twisted = input("\nWas the trunk twisted? (y/n): ").strip().lower()
            if twisted in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            side_bent = input("Was the trunk side bent? (y/n): ").strip().lower()
            if side_bent in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        return twisted == "y", side_bent == "y"

    # LEGS ---------------------------------------------------------------------------------------------------------------
    ## Legs Raised
    def calculate_ankles_y_diff(self, landmarks, frame_width, frame_height):
        """Calculate normalized left and right ankles y-difference as leg raised metric"""

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_ankle = scale(landmarks[27])
        right_ankle = scale(landmarks[28])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])

        ankles_y_diff = abs(left_ankle["y"] - right_ankle["y"])

        # Normalize by leg length (hip midpoint to ankle midpoint y-distance)
        hip_mid_y = (left_hip["y"] + right_hip["y"]) / 2
        ankle_mid_y = (left_ankle["y"] + right_ankle["y"]) / 2
        leg_length = abs(hip_mid_y - ankle_mid_y) + 1e-8

        return ankles_y_diff / leg_length

    def calculate_leg_raised(self, landmarks, frame_width, frame_height, baseline_mean):
        """Returns True if leg raised detected"""
        if baseline_mean is None:
            return False
        current = self.calculate_ankles_y_diff(landmarks, frame_width, frame_height)
        return bool(current > 2 * baseline_mean)

    def calculate_leg_reba_score(self, leg_raised, left_knee_angle, right_knee_angle):
        base_score = 2 if leg_raised else 1
        left_knee_score = self.calculate_knee_reba_score(left_knee_angle)
        right_knee_score = self.calculate_knee_reba_score(right_knee_angle)
        knee_score = max(left_knee_score, right_knee_score)
        return base_score + knee_score

    ## Knee Angle
    def calculate_knee_flexion_angle(self, landmarks, frame_width, frame_height):
        """
        Calculate knee flexion angle for both knees using hip-knee-ankle.
        Returns tuple (left_flexion, right_flexion) in degrees
        """

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])
        left_knee = scale(landmarks[25])
        right_knee = scale(landmarks[26])
        left_ankle = scale(landmarks[27])
        right_ankle = scale(landmarks[28])

        def compute_flexion(hip, knee, ankle):
            thigh_vec = np.array([hip["x"] - knee["x"], hip["y"] - knee["y"]])
            shin_vec = np.array([ankle["x"] - knee["x"], ankle["y"] - knee["y"]])
            cos_angle = np.dot(thigh_vec, shin_vec) / (
                np.linalg.norm(thigh_vec) * np.linalg.norm(shin_vec) + 1e-8
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            return 180 - angle

        left_flexion = compute_flexion(left_hip, left_knee, left_ankle)
        right_flexion = compute_flexion(right_hip, right_knee, right_ankle)

        """
        # Debug
        # Normalised coordinates
        print(f"[Knee] L Hip (norm):   ({landmarks[23]['x']:.4f}, {landmarks[23]['y']:.4f})")
        print(f"[Knee] R Hip (norm):   ({landmarks[24]['x']:.4f}, {landmarks[24]['y']:.4f})")
        print(f"[Knee] L Knee (norm):  ({landmarks[25]['x']:.4f}, {landmarks[25]['y']:.4f})")
        print(f"[Knee] R Knee (norm):  ({landmarks[26]['x']:.4f}, {landmarks[26]['y']:.4f})")
        print(f"[Knee] L Ankle (norm): ({landmarks[27]['x']:.4f}, {landmarks[27]['y']:.4f})")
        print(f"[Knee] R Ankle (norm): ({landmarks[28]['x']:.4f}, {landmarks[28]['y']:.4f})")

        # Scaled coordinates
        print(f"[Knee] L Hip:   ({left_hip['x']:.1f}, {left_hip['y']:.1f})")
        print(f"[Knee] R Hip:   ({right_hip['x']:.1f}, {right_hip['y']:.1f})")
        print(f"[Knee] L Knee:  ({left_knee['x']:.1f}, {left_knee['y']:.1f})")
        print(f"[Knee] R Knee:  ({right_knee['x']:.1f}, {right_knee['y']:.1f})")
        print(f"[Knee] L Ankle: ({left_ankle['x']:.1f}, {left_ankle['y']:.1f})")
        print(f"[Knee] R Ankle: ({right_ankle['x']:.1f}, {right_ankle['y']:.1f})")

        # Vectors and angles
        left_thigh_vec = np.array([left_hip['x'] - left_knee['x'], left_hip['y'] - left_knee['y']])
        left_shin_vec = np.array([left_ankle['x'] - left_knee['x'], left_ankle['y'] - left_knee['y']])
        right_thigh_vec = np.array([right_hip['x'] - right_knee['x'], right_hip['y'] - right_knee['y']])
        right_shin_vec = np.array([right_ankle['x'] - right_knee['x'], right_ankle['y'] - right_knee['y']])

        print(f"[Knee] L Thigh Vec: ({left_thigh_vec[0]:.1f}, {left_thigh_vec[1]:.1f})")
        print(f"[Knee] L Shin Vec:  ({left_shin_vec[0]:.1f}, {left_shin_vec[1]:.1f})")
        print(f"[Knee] R Thigh Vec: ({right_thigh_vec[0]:.1f}, {right_thigh_vec[1]:.1f})")
        print(f"[Knee] R Shin Vec:  ({right_shin_vec[0]:.1f}, {right_shin_vec[1]:.1f})")
        print(f"[Knee] L Flexion: {left_flexion:.2f} deg")
        print(f"[Knee] R Flexion: {right_flexion:.2f} deg")
        print("-" * 40)
        """

        # Cap Knee flexion: 0 to 180
        left_flexion = float(np.clip(left_flexion, 0, 180))
        right_flexion = float(np.clip(right_flexion, 0, 180))

        return left_flexion, right_flexion

    def calculate_knee_reba_score(self, knee_flexion_angle):
        """REBA knee flexion score"""
        if knee_flexion_angle < 30:
            return 0
        elif knee_flexion_angle <= 60:
            return 1
        else:
            return 2

    # SECTION A MISC -------------------------------------------------------------------------------------------------------
    def calculate_reba_table_a(self, neck_score, trunk_score, legs_score):
        """Lookup REBA Table A score using neck, trunk, legs scores"""
        table_a = {
            (1, 1): [1, 2, 2, 3, 4],
            (1, 2): [2, 3, 4, 5, 6],
            (1, 3): [3, 4, 5, 6, 7],
            (1, 4): [4, 5, 6, 7, 8],
            (2, 1): [1, 3, 4, 5, 6],
            (2, 2): [2, 4, 5, 6, 7],
            (2, 3): [3, 5, 6, 7, 8],
            (2, 4): [4, 6, 7, 8, 9],
            (3, 1): [3, 4, 5, 6, 7],
            (3, 2): [3, 5, 6, 7, 8],
            (3, 3): [5, 6, 7, 8, 9],
            (3, 4): [6, 7, 8, 9, 9],
        }

        if any(np.isnan(x) for x in [neck_score, trunk_score, legs_score]):
            return float("nan")

        neck = int(neck_score)
        trunk = int(trunk_score)
        legs = int(legs_score)

        neck = np.clip(neck, 1, 3)
        trunk = np.clip(trunk, 1, 5)
        legs = np.clip(legs, 1, 4)

        return table_a[(neck, legs)][trunk - 1]

    def get_force_load_score(self):
        """Prompt user for force/load input and return score"""
        while True:
            try:
                weight = float(
                    input("\nEnter weight of object subject is holding (kg): ")
                )
                break
            except ValueError:
                print("Invalid input, please enter a number.")

        if weight < 5:
            force_score = 0
        elif weight <= 10:
            force_score = 1
        else:
            force_score = 2

        while True:
            shock = (
                input(
                    "Was there sudden/shock loading or rapid build up of force? (y/n): "
                )
                .strip()
                .lower()
            )
            if shock in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")

        if shock == "y":
            force_score += 1

        print(f"Force/Load Score: {force_score}")
        return force_score

    def calculate_score_a(self, table_a_score, force_load_score):
        """Calculate Score A = Table A Score + Force/Load Score"""
        if np.isnan(table_a_score):
            return float("nan")
        return table_a_score + force_load_score

    ################################################# REBA SECTION B #######################################################
    # UPPER ARM -------------------------------------------------------------------------------------------------------
    ## Upper Arm Angle
    def calculate_upper_arm_angle(self, landmarks, frame_width, frame_height):
        """
        Calculate upper arm flexion/extension angle relative to trunk for both arms.
        Returns tuple (left_angle, right_angle) in degrees (positive = flexion, negative = extension)
        """

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_hip = scale(landmarks[23])
        right_hip = scale(landmarks[24])
        left_elbow = scale(landmarks[13])
        right_elbow = scale(landmarks[14])

        shoulder_mid = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) / 2,
                (left_shoulder["y"] + right_shoulder["y"]) / 2,
            ]
        )
        hip_mid = np.array(
            [(left_hip["x"] + right_hip["x"]) / 2, (left_hip["y"] + right_hip["y"]) / 2]
        )

        trunk_vec = hip_mid - shoulder_mid

        def compute_angle(shoulder, elbow, facing_direction):
            shoulder_pt = np.array([shoulder["x"], shoulder["y"]])
            elbow_pt = np.array([elbow["x"], elbow["y"]])
            arm_vec = elbow_pt - shoulder_pt
            cos_angle = np.dot(trunk_vec, arm_vec) / (
                np.linalg.norm(trunk_vec) * np.linalg.norm(arm_vec) + 1e-8
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            cross_z = trunk_vec[0] * arm_vec[1] - trunk_vec[1] * arm_vec[0]
            if facing_direction * cross_z < 0:
                angle = -angle
            return angle

        left_angle = compute_angle(left_shoulder, left_elbow, facing_direction=-1)
        right_angle = compute_angle(right_shoulder, right_elbow, facing_direction=-1)

        """
        # Debug
        # Normalised coordinates
        print(f"[Upper Arm] L Shoulder (norm): ({landmarks[11]['x']:.4f}, {landmarks[11]['y']:.4f})")
        print(f"[Upper Arm] R Shoulder (norm): ({landmarks[12]['x']:.4f}, {landmarks[12]['y']:.4f})")
        print(f"[Upper Arm] L Hip (norm):      ({landmarks[23]['x']:.4f}, {landmarks[23]['y']:.4f})")
        print(f"[Upper Arm] R Hip (norm):      ({landmarks[24]['x']:.4f}, {landmarks[24]['y']:.4f})")
        print(f"[Upper Arm] L Elbow (norm):    ({landmarks[13]['x']:.4f}, {landmarks[13]['y']:.4f})")
        print(f"[Upper Arm] R Elbow (norm):    ({landmarks[14]['x']:.4f}, {landmarks[14]['y']:.4f})")

        # Scaled coordinates
        print(f"[Upper Arm] L Shoulder: ({left_shoulder['x']:.1f}, {left_shoulder['y']:.1f})")
        print(f"[Upper Arm] R Shoulder: ({right_shoulder['x']:.1f}, {right_shoulder['y']:.1f})")
        print(f"[Upper Arm] L Hip:      ({left_hip['x']:.1f}, {left_hip['y']:.1f})")
        print(f"[Upper Arm] R Hip:      ({right_hip['x']:.1f}, {right_hip['y']:.1f})")
        print(f"[Upper Arm] L Elbow:    ({left_elbow['x']:.1f}, {left_elbow['y']:.1f})")
        print(f"[Upper Arm] R Elbow:    ({right_elbow['x']:.1f}, {right_elbow['y']:.1f})")

        # Midpoints and vectors
        print(f"[Upper Arm] Shoulder Mid: ({shoulder_mid[0]:.1f}, {shoulder_mid[1]:.1f})")
        print(f"[Upper Arm] Hip Mid:      ({hip_mid[0]:.1f}, {hip_mid[1]:.1f})")
        print(f"[Upper Arm] Trunk Vec:    ({trunk_vec[0]:.1f}, {trunk_vec[1]:.1f})")

        left_arm_vec = np.array([left_elbow['x'] - left_shoulder['x'], left_elbow['y'] - left_shoulder['y']])
        right_arm_vec = np.array([right_elbow['x'] - right_shoulder['x'], right_elbow['y'] - right_shoulder['y']])
        print(f"[Upper Arm] L Arm Vec: ({left_arm_vec[0]:.1f}, {left_arm_vec[1]:.1f})")
        print(f"[Upper Arm] R Arm Vec: ({right_arm_vec[0]:.1f}, {right_arm_vec[1]:.1f})")

        print(f"[Upper Arm] L Angle: {left_angle:.2f} deg")
        print(f"[Upper Arm] R Angle: {right_angle:.2f} deg")
        print("-" * 40)
        """

        # Cap Upper Arm flexion: -50 to 180
        left_angle = float(np.clip(left_angle, -50, 180))
        right_angle = float(np.clip(right_angle, -50, 180))

        return left_angle, right_angle

    def calculate_upper_arm_reba_score(self, angle):
        """REBA upper arm score based on flexion/extension angle"""
        if np.isnan(angle):
            return float("nan")
        if -20 <= angle <= 20:
            return 1
        elif angle < -20 or (20 < angle <= 45):
            return 2
        elif 45 < angle <= 90:
            return 3
        else:  # > 90
            return 4

    ## Upper Arm Shoulder Raised & Upper Arm Abducted
    def get_upper_arm_adjustments(self):
        """Prompt user for shoulder raised and upper arm abducted"""
        while True:
            raised = input("\nWas the shoulder raised? (y/n): ").strip().lower()
            if raised in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            abducted = input("Was the upper arm abducted? (y/n): ").strip().lower()
            if abducted in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        return raised == "y", abducted == "y"

    ## Upper Arm Supported/Leaning
    def get_arm_supported_score(self):
        """Prompt user for arm supported/leaning input and return adjustment score"""
        while True:
            supported = input("\nIs the arm supported? (y/n): ").strip().lower()
            if supported in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")

        while True:
            leaning = input("Is the person leaning? (y/n): ").strip().lower()
            if leaning in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")

        if supported == "y" or leaning == "y":
            adjustment = -1
        else:
            adjustment = 0

        print(f"Arm Supported/Leaning Adjustment: {adjustment}")
        return adjustment

    # LOWER ARM ------------------------------------------------------------------------------------------------------------
    def calculate_lower_arm_angle(self, landmarks, frame_width, frame_height):
        """
        Calculate lower arm flexion angle at elbow joint for both arms.
        Returns tuple (left_angle, right_angle) in degrees.
        0 = fully extended, 180 = fully flexed.
        """

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_shoulder = scale(landmarks[11])
        right_shoulder = scale(landmarks[12])
        left_elbow = scale(landmarks[13])
        right_elbow = scale(landmarks[14])
        left_wrist = scale(landmarks[15])
        right_wrist = scale(landmarks[16])

        def compute_angle(shoulder, elbow, wrist):
            shoulder_pt = np.array([shoulder["x"], shoulder["y"]])
            elbow_pt = np.array([elbow["x"], elbow["y"]])
            wrist_pt = np.array([wrist["x"], wrist["y"]])

            upper_arm_vec = elbow_pt - shoulder_pt
            lower_arm_vec = wrist_pt - elbow_pt

            cos_angle = np.dot(upper_arm_vec, lower_arm_vec) / (
                np.linalg.norm(upper_arm_vec) * np.linalg.norm(lower_arm_vec) + 1e-8
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            return angle

        left_angle = compute_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = compute_angle(right_shoulder, right_elbow, right_wrist)

        """
        # Debug
        # Normalised coordinates
        print(f"[Lower Arm] L Shoulder (norm): ({landmarks[11]['x']:.4f}, {landmarks[11]['y']:.4f})")
        print(f"[Lower Arm] R Shoulder (norm): ({landmarks[12]['x']:.4f}, {landmarks[12]['y']:.4f})")
        print(f"[Lower Arm] L Elbow (norm):    ({landmarks[13]['x']:.4f}, {landmarks[13]['y']:.4f})")
        print(f"[Lower Arm] R Elbow (norm):    ({landmarks[14]['x']:.4f}, {landmarks[14]['y']:.4f})")
        print(f"[Lower Arm] L Wrist (norm):    ({landmarks[15]['x']:.4f}, {landmarks[15]['y']:.4f})")
        print(f"[Lower Arm] R Wrist (norm):    ({landmarks[16]['x']:.4f}, {landmarks[16]['y']:.4f})")

        # Scaled coordinates
        print(f"[Lower Arm] L Shoulder: ({left_shoulder['x']:.1f}, {left_shoulder['y']:.1f})")
        print(f"[Lower Arm] R Shoulder: ({right_shoulder['x']:.1f}, {right_shoulder['y']:.1f})")
        print(f"[Lower Arm] L Elbow:    ({left_elbow['x']:.1f}, {left_elbow['y']:.1f})")
        print(f"[Lower Arm] R Elbow:    ({right_elbow['x']:.1f}, {right_elbow['y']:.1f})")
        print(f"[Lower Arm] L Wrist:    ({left_wrist['x']:.1f}, {left_wrist['y']:.1f})")
        print(f"[Lower Arm] R Wrist:    ({right_wrist['x']:.1f}, {right_wrist['y']:.1f})")

        # Vectors and angles
        left_upper_vec = np.array([left_elbow['x'] - left_shoulder['x'], left_elbow['y'] - left_shoulder['y']])
        left_lower_vec = np.array([left_wrist['x'] - left_elbow['x'], left_wrist['y'] - left_elbow['y']])
        right_upper_vec = np.array([right_elbow['x'] - right_shoulder['x'], right_elbow['y'] - right_shoulder['y']])
        right_lower_vec = np.array([right_wrist['x'] - right_elbow['x'], right_wrist['y'] - right_elbow['y']])
        print(f"[Lower Arm] L Upper Vec: ({left_upper_vec[0]:.1f}, {left_upper_vec[1]:.1f})")
        print(f"[Lower Arm] L Lower Vec: ({left_lower_vec[0]:.1f}, {left_lower_vec[1]:.1f})")
        print(f"[Lower Arm] R Upper Vec: ({right_upper_vec[0]:.1f}, {right_upper_vec[1]:.1f})")
        print(f"[Lower Arm] R Lower Vec: ({right_lower_vec[0]:.1f}, {right_lower_vec[1]:.1f})")
        print(f"[Lower Arm] L Angle: {left_angle:.2f} deg")
        print(f"[Lower Arm] R Angle: {right_angle:.2f} deg")
        print("-" * 40)
        """

        # Cap Elbow flexion: 0 to 180
        left_angle = float(np.clip(left_angle, 0, 180))
        right_angle = float(np.clip(right_angle, 0, 180))

        return left_angle, right_angle

    def calculate_lower_arm_reba_score(self, angle):
        """REBA lower arm score"""
        if np.isnan(angle):
            return float("nan")
        if 60 <= angle <= 100:
            return 1
        else:
            return 2

    # WRIST ----------------------------------------------------------------------------------------------------------------
    ## Wrist Angle
    def calculate_wrist_angle(
        self, landmarks, facing_direction, frame_width, frame_height
    ):
        """
        Calculate wrist flexion/extension angle for both wrists.
        Returns tuple (left_angle, right_angle) in degrees.
        Positive = flexion, negative = extension.
        """

        def scale(lm):
            return {"x": lm["x"] * frame_width, "y": lm["y"] * frame_height}

        left_elbow = scale(landmarks[13])
        right_elbow = scale(landmarks[14])
        left_wrist = scale(landmarks[15])
        right_wrist = scale(landmarks[16])
        left_pinky = scale(landmarks[17])
        right_pinky = scale(landmarks[18])
        left_index = scale(landmarks[19])
        right_index = scale(landmarks[20])

        def compute_angle(elbow, wrist, pinky, index, facing):
            elbow_pt = np.array([elbow["x"], elbow["y"]])
            wrist_pt = np.array([wrist["x"], wrist["y"]])
            pinky_pt = np.array([pinky["x"], pinky["y"]])
            index_pt = np.array([index["x"], index["y"]])

            finger_mid = (pinky_pt + index_pt) / 2
            forearm_vec = wrist_pt - elbow_pt
            hand_vec = finger_mid - wrist_pt

            cos_angle = np.dot(forearm_vec, hand_vec) / (
                np.linalg.norm(forearm_vec) * np.linalg.norm(hand_vec) + 1e-8
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            cross_z = forearm_vec[0] * hand_vec[1] - forearm_vec[1] * hand_vec[0]
            if facing * cross_z < 0:
                angle = -angle
            return angle

        left_angle = compute_angle(
            left_elbow, left_wrist, left_pinky, left_index, facing_direction
        )
        right_angle = compute_angle(
            right_elbow, right_wrist, right_pinky, right_index, facing_direction
        )

        """
        # Debug
        # Normalised coordinates
        print(f"[Wrist] L Elbow (norm):  ({landmarks[13]['x']:.4f}, {landmarks[13]['y']:.4f})")
        print(f"[Wrist] R Elbow (norm):  ({landmarks[14]['x']:.4f}, {landmarks[14]['y']:.4f})")
        print(f"[Wrist] L Wrist (norm):  ({landmarks[15]['x']:.4f}, {landmarks[15]['y']:.4f})")
        print(f"[Wrist] R Wrist (norm):  ({landmarks[16]['x']:.4f}, {landmarks[16]['y']:.4f})")
        print(f"[Wrist] L Pinky (norm):  ({landmarks[17]['x']:.4f}, {landmarks[17]['y']:.4f})")
        print(f"[Wrist] R Pinky (norm):  ({landmarks[18]['x']:.4f}, {landmarks[18]['y']:.4f})")
        print(f"[Wrist] L Index (norm):  ({landmarks[19]['x']:.4f}, {landmarks[19]['y']:.4f})")
        print(f"[Wrist] R Index (norm):  ({landmarks[20]['x']:.4f}, {landmarks[20]['y']:.4f})")

        # Scaled coordinates
        print(f"[Wrist] L Elbow:  ({left_elbow['x']:.1f}, {left_elbow['y']:.1f})")
        print(f"[Wrist] R Elbow:  ({right_elbow['x']:.1f}, {right_elbow['y']:.1f})")
        print(f"[Wrist] L Wrist:  ({left_wrist['x']:.1f}, {left_wrist['y']:.1f})")
        print(f"[Wrist] R Wrist:  ({right_wrist['x']:.1f}, {right_wrist['y']:.1f})")
        print(f"[Wrist] L Pinky:  ({left_pinky['x']:.1f}, {left_pinky['y']:.1f})")
        print(f"[Wrist] R Pinky:  ({right_pinky['x']:.1f}, {right_pinky['y']:.1f})")
        print(f"[Wrist] L Index:  ({left_index['x']:.1f}, {left_index['y']:.1f})")
        print(f"[Wrist] R Index:  ({right_index['x']:.1f}, {right_index['y']:.1f})")

        # Vectors and angles
        left_forearm_vec = np.array([left_wrist['x'] - left_elbow['x'], left_wrist['y'] - left_elbow['y']])
        right_forearm_vec = np.array([right_wrist['x'] - right_elbow['x'], right_wrist['y'] - right_elbow['y']])
        left_finger_mid = np.array([(left_pinky['x'] + left_index['x']) / 2, (left_pinky['y'] + left_index['y']) / 2])
        right_finger_mid = np.array(
            [(right_pinky['x'] + right_index['x']) / 2, (right_pinky['y'] + right_index['y']) / 2])
        left_hand_vec = left_finger_mid - np.array([left_wrist['x'], left_wrist['y']])
        right_hand_vec = right_finger_mid - np.array([right_wrist['x'], right_wrist['y']])
        print(f"[Wrist] L Forearm Vec: ({left_forearm_vec[0]:.1f}, {left_forearm_vec[1]:.1f})")
        print(f"[Wrist] R Forearm Vec: ({right_forearm_vec[0]:.1f}, {right_forearm_vec[1]:.1f})")
        print(f"[Wrist] L Hand Vec:    ({left_hand_vec[0]:.1f}, {left_hand_vec[1]:.1f})")
        print(f"[Wrist] R Hand Vec:    ({right_hand_vec[0]:.1f}, {right_hand_vec[1]:.1f})")
        print(f"[Wrist] L Finger Mid:  ({left_finger_mid[0]:.1f}, {left_finger_mid[1]:.1f})")
        print(f"[Wrist] R Finger Mid:  ({right_finger_mid[0]:.1f}, {right_finger_mid[1]:.1f})")
        print(f"[Wrist] facing_direction: {facing_direction}")
        print(f"[Wrist] L Angle: {left_angle:.2f} deg")
        print(f"[Wrist] R Angle: {right_angle:.2f} deg")
        print("-" * 40)
        """

        # Cap Wrist flexion: -60 to 60
        left_angle = float(np.clip(left_angle, -60, 60))
        right_angle = float(np.clip(right_angle, -60, 60))

        return left_angle, right_angle

    def calculate_wrist_reba_score(self, angle):
        """REBA wrist score"""
        if np.isnan(angle):
            return float("nan")
        if abs(angle) <= 15:
            return 1
        else:
            return 2

    ## Wrist Bent from Midline / Twisted
    def get_wrist_adjustments(self):
        """Prompt user for wrist bent and twisted per side"""
        while True:
            left_bent = (
                input("\nWas the left wrist bent from midline? (y/n): ").strip().lower()
            )
            if left_bent in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            right_bent = (
                input("Was the right wrist bent from midline? (y/n): ").strip().lower()
            )
            if right_bent in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            left_twisted = input("Was the left wrist twisted? (y/n): ").strip().lower()
            if left_twisted in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        while True:
            right_twisted = (
                input("Was the right wrist twisted? (y/n): ").strip().lower()
            )
            if right_twisted in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        return (
            left_bent == "y",
            right_bent == "y",
            left_twisted == "y",
            right_twisted == "y",
        )

    # SECTION B MISC -------------------------------------------------------------------------------------------------------
    def calculate_reba_table_b(self, lower_arm_score, upper_arm_score, wrist_score):
        """Lookup REBA Table B score using upper arm, lower arm, wrist scores"""
        table_b = {
            (1, 1, 1): 1,
            (1, 1, 2): 2,
            (1, 1, 3): 2,
            (1, 2, 1): 1,
            (1, 2, 2): 2,
            (1, 2, 3): 3,
            (1, 3, 1): 3,
            (1, 3, 2): 4,
            (1, 3, 3): 5,
            (1, 4, 1): 4,
            (1, 4, 2): 5,
            (1, 4, 3): 5,
            (1, 5, 1): 6,
            (1, 5, 2): 7,
            (1, 5, 3): 8,
            (1, 6, 1): 7,
            (1, 6, 2): 8,
            (1, 6, 3): 8,
            (2, 1, 1): 1,
            (2, 1, 2): 2,
            (2, 1, 3): 3,
            (2, 2, 1): 2,
            (2, 2, 2): 3,
            (2, 2, 3): 4,
            (2, 3, 1): 4,
            (2, 3, 2): 5,
            (2, 3, 3): 5,
            (2, 4, 1): 5,
            (2, 4, 2): 6,
            (2, 4, 3): 7,
            (2, 5, 1): 7,
            (2, 5, 2): 8,
            (2, 5, 3): 8,
            (2, 6, 1): 8,
            (2, 6, 2): 9,
            (2, 6, 3): 9,
        }

        if any(np.isnan(x) for x in [upper_arm_score, lower_arm_score, wrist_score]):
            return float("nan")

        lower_arm = int(lower_arm_score)
        upper_arm = int(upper_arm_score)
        wrist = int(wrist_score)

        lower_arm = np.clip(lower_arm, 1, 2)
        upper_arm = np.clip(upper_arm, 1, 6)
        wrist = np.clip(wrist, 1, 3)

        return table_b[(lower_arm, upper_arm, wrist)]

    def get_coupling_score(self):
        """Prompt user for coupling score"""
        print("\nSelect coupling type:")
        print("0: Well fitting handle and mid range power grip (Good)")
        print(
            "1: Acceptable but not ideal hand hold or coupling acceptable with another body part (Fair)"
        )
        print("2: Hand hold not acceptable but possible (Poor)")
        print("3: No handles, awkward, unsafe with any body part (Unacceptable)")
        while True:
            try:
                score = int(input("Enter coupling score (0/1/2/3): ").strip())
                if score in [0, 1, 2, 3]:
                    break
                print("Invalid input, please enter 0, 1, 2, or 3.")
            except ValueError:
                print("Invalid input, please enter 0, 1, 2, or 3.")
        print(f"Coupling Score: {score}")
        return score

    def calculate_score_b(self, table_b_score, coupling_score):
        """Calculate Score B = Table B Score + Coupling Score"""
        if np.isnan(table_b_score):
            return float("nan")
        return table_b_score + coupling_score

    def calculate_reba_table_c(self, score_a, score_b):
        """Lookup REBA Table C score using Score A and Score B"""
        table_c = {
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 2,
            (1, 5): 3,
            (1, 6): 3,
            (1, 7): 4,
            (1, 8): 5,
            (1, 9): 6,
            (1, 10): 7,
            (1, 11): 7,
            (1, 12): 7,
            (2, 1): 1,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 3,
            (2, 5): 4,
            (2, 6): 4,
            (2, 7): 5,
            (2, 8): 6,
            (2, 9): 6,
            (2, 10): 7,
            (2, 11): 7,
            (2, 12): 8,
            (3, 1): 2,
            (3, 2): 3,
            (3, 3): 3,
            (3, 4): 3,
            (3, 5): 4,
            (3, 6): 5,
            (3, 7): 6,
            (3, 8): 7,
            (3, 9): 7,
            (3, 10): 8,
            (3, 11): 8,
            (3, 12): 8,
            (4, 1): 3,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 5,
            (4, 6): 6,
            (4, 7): 7,
            (4, 8): 8,
            (4, 9): 8,
            (4, 10): 9,
            (4, 11): 9,
            (4, 12): 9,
            (5, 1): 4,
            (5, 2): 4,
            (5, 3): 4,
            (5, 4): 5,
            (5, 5): 6,
            (5, 6): 7,
            (5, 7): 8,
            (5, 8): 8,
            (5, 9): 9,
            (5, 10): 9,
            (5, 11): 9,
            (5, 12): 9,
            (6, 1): 6,
            (6, 2): 6,
            (6, 3): 6,
            (6, 4): 7,
            (6, 5): 8,
            (6, 6): 8,
            (6, 7): 9,
            (6, 8): 9,
            (6, 9): 10,
            (6, 10): 10,
            (6, 11): 10,
            (6, 12): 10,
            (7, 1): 7,
            (7, 2): 7,
            (7, 3): 7,
            (7, 4): 8,
            (7, 5): 9,
            (7, 6): 9,
            (7, 7): 9,
            (7, 8): 10,
            (7, 9): 10,
            (7, 10): 11,
            (7, 11): 11,
            (7, 12): 11,
            (8, 1): 8,
            (8, 2): 8,
            (8, 3): 8,
            (8, 4): 9,
            (8, 5): 10,
            (8, 6): 10,
            (8, 7): 10,
            (8, 8): 10,
            (8, 9): 10,
            (8, 10): 11,
            (8, 11): 11,
            (8, 12): 11,
            (9, 1): 9,
            (9, 2): 9,
            (9, 3): 9,
            (9, 4): 10,
            (9, 5): 10,
            (9, 6): 10,
            (9, 7): 11,
            (9, 8): 11,
            (9, 9): 11,
            (9, 10): 12,
            (9, 11): 12,
            (9, 12): 12,
            (10, 1): 10,
            (10, 2): 10,
            (10, 3): 10,
            (10, 4): 11,
            (10, 5): 11,
            (10, 6): 11,
            (10, 7): 11,
            (10, 8): 12,
            (10, 9): 12,
            (10, 10): 12,
            (10, 11): 12,
            (10, 12): 12,
            (11, 1): 11,
            (11, 2): 11,
            (11, 3): 11,
            (11, 4): 11,
            (11, 5): 12,
            (11, 6): 12,
            (11, 7): 12,
            (11, 8): 12,
            (11, 9): 12,
            (11, 10): 12,
            (11, 11): 12,
            (11, 12): 12,
            (12, 1): 12,
            (12, 2): 12,
            (12, 3): 12,
            (12, 4): 12,
            (12, 5): 12,
            (12, 6): 12,
            (12, 7): 12,
            (12, 8): 12,
            (12, 9): 12,
            (12, 10): 12,
            (12, 11): 12,
            (12, 12): 12,
        }

        if any(np.isnan(x) for x in [score_a, score_b]):
            return float("nan")

        a = int(np.clip(score_a, 1, 12))
        b = int(np.clip(score_b, 1, 12))

        return table_c[(a, b)]

    def get_activity_score(self):
        """Prompt user for activity score"""
        score = 0

        while True:
            static = (
                input(
                    "\nAre one or more body parts static, held for longer than 1 minute? (y/n): "
                )
                .strip()
                .lower()
            )
            if static in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        if static == "y":
            score += 1

        while True:
            repeated = (
                input(
                    "Are there repeated small range actions, repeated more than 4 times per minute? (y/n): "
                )
                .strip()
                .lower()
            )
            if repeated in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        if repeated == "y":
            score += 1

        while True:
            rapid = (
                input(
                    "Does the action cause rapid large range changes in posture or an unstable base? (y/n): "
                )
                .strip()
                .lower()
            )
            if rapid in ["y", "n"]:
                break
            print("Invalid input, please enter y or n.")
        if rapid == "y":
            score += 1

        print(f"Activity Score: {score}")
        return score

    # NATURAL ANGLES -------------------------------------------------------------------------------------------------------
    def calculate_natural_angles(self, corrected_results, mp_result0):
        """Calculate natural angles by subtracting neutral baselines from corrected (or raw) angles"""

        def get_angle(corr_key, mp_key):
            if (
                corrected_results
                and corr_key in corrected_results
                and not np.isnan(corrected_results[corr_key])
            ):
                return corrected_results[corr_key]
            v = mp_result0.get(mp_key, float("nan"))
            return v if v is not None else float("nan")

        def subtract(angle, baseline):
            if np.isnan(angle) or baseline is None:
                return float("nan")
            return angle - baseline

        neck_angle_natural = subtract(
            get_angle("neck_angle_corrected", "neck_angle"), self.neck_baseline_angle0
        )
        neck_reba_natural = (
            self.calculate_neck_reba_score(neck_angle_natural)
            if not np.isnan(neck_angle_natural)
            else float("nan")
        )

        trunk_angle_natural = subtract(
            get_angle("trunk_angle_corrected", "trunk_angle"),
            self.trunk_baseline_angle0,
        )
        trunk_reba_natural = (
            self.calculate_trunk_reba_score(trunk_angle_natural, 0.0)
            if not np.isnan(trunk_angle_natural)
            else float("nan")
        )

        left_knee_natural = subtract(
            get_angle("left_knee_angle_corrected", "left_knee_angle"),
            self.left_knee_baseline_angle0,
        )
        right_knee_natural = subtract(
            get_angle("right_knee_angle_corrected", "right_knee_angle"),
            self.right_knee_baseline_angle0,
        )
        left_knee_reba_natural = (
            self.calculate_knee_reba_score(left_knee_natural)
            if not np.isnan(left_knee_natural)
            else float("nan")
        )
        right_knee_reba_natural = (
            self.calculate_knee_reba_score(right_knee_natural)
            if not np.isnan(right_knee_natural)
            else float("nan")
        )
        leg_reba_natural = (
            self.calculate_leg_reba_score(
                get_angle("leg_raised_corrected", "leg_raised"),
                left_knee_natural,
                right_knee_natural,
            )
            if not np.isnan(left_knee_natural)
            else float("nan")
        )

        left_ua_natural = subtract(
            get_angle("left_upper_arm_angle_corrected", "left_upper_arm_angle"),
            self.left_upper_arm_baseline_angle0,
        )
        right_ua_natural = subtract(
            get_angle("right_upper_arm_angle_corrected", "right_upper_arm_angle"),
            self.right_upper_arm_baseline_angle0,
        )
        left_ua_reba_natural = (
            self.calculate_upper_arm_reba_score(left_ua_natural)
            if not np.isnan(left_ua_natural)
            else float("nan")
        )
        right_ua_reba_natural = (
            self.calculate_upper_arm_reba_score(right_ua_natural)
            if not np.isnan(right_ua_natural)
            else float("nan")
        )
        ua_reba_natural = (
            max(left_ua_reba_natural, right_ua_reba_natural)
            if not any(np.isnan([left_ua_reba_natural, right_ua_reba_natural]))
            else float("nan")
        )

        left_la_natural = subtract(
            get_angle("left_lower_arm_angle_corrected", "left_lower_arm_angle"),
            self.left_lower_arm_baseline_angle0,
        )
        right_la_natural = subtract(
            get_angle("right_lower_arm_angle_corrected", "right_lower_arm_angle"),
            self.right_lower_arm_baseline_angle0,
        )
        left_la_reba_natural = (
            self.calculate_lower_arm_reba_score(left_la_natural)
            if not np.isnan(left_la_natural)
            else float("nan")
        )
        right_la_reba_natural = (
            self.calculate_lower_arm_reba_score(right_la_natural)
            if not np.isnan(right_la_natural)
            else float("nan")
        )
        la_reba_natural = (
            max(left_la_reba_natural, right_la_reba_natural)
            if not any(np.isnan([left_la_reba_natural, right_la_reba_natural]))
            else float("nan")
        )

        left_wrist_natural = subtract(
            get_angle("left_wrist_angle_corrected", "left_wrist_angle"),
            self.left_wrist_baseline_angle0,
        )
        right_wrist_natural = subtract(
            get_angle("right_wrist_angle_corrected", "right_wrist_angle"),
            self.right_wrist_baseline_angle0,
        )
        left_wrist_reba_natural = (
            self.calculate_wrist_reba_score(left_wrist_natural)
            if not np.isnan(left_wrist_natural)
            else float("nan")
        )
        right_wrist_reba_natural = (
            self.calculate_wrist_reba_score(right_wrist_natural)
            if not np.isnan(right_wrist_natural)
            else float("nan")
        )
        wrist_reba_natural = (
            max(left_wrist_reba_natural, right_wrist_reba_natural)
            if not any(np.isnan([left_wrist_reba_natural, right_wrist_reba_natural]))
            else float("nan")
        )

        return {
            "neck_angle_natural": neck_angle_natural,
            "neck_reba_score_natural": neck_reba_natural,
            "trunk_angle_natural": trunk_angle_natural,
            "trunk_reba_score_natural": trunk_reba_natural,
            "left_knee_angle_natural": left_knee_natural,
            "right_knee_angle_natural": right_knee_natural,
            "leg_reba_score_natural": leg_reba_natural,
            "left_upper_arm_angle_natural": left_ua_natural,
            "right_upper_arm_angle_natural": right_ua_natural,
            "upper_arm_reba_score_natural": ua_reba_natural,
            "left_lower_arm_angle_natural": left_la_natural,
            "right_lower_arm_angle_natural": right_la_natural,
            "lower_arm_reba_score_natural": la_reba_natural,
            "left_wrist_angle_natural": left_wrist_natural,
            "right_wrist_angle_natural": right_wrist_natural,
            "wrist_reba_score_natural": wrist_reba_natural,
        }

    ####################################################### PROCESSING #####################################################

    def process_mediapipe_frame(
        self,
        frame,
        pose_detector,
        previous_ema,
        previous_landmarks,
        stability_history,
        facing_direction=None,
        frame_width=None,
        frame_height=None,
        camera_role="side",
    ):
        """Process frame with MediaPipe and return results"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(
                    {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )

            smoothed_landmarks = self.apply_ema(landmarks, previous_ema)
            stability_scores = self.calculate_stability(
                smoothed_landmarks, previous_landmarks, stability_history
            )

            valid_count = 0
            for lm in smoothed_landmarks:
                if lm["visibility"] >= 0.5:
                    valid_count += 1

            valid_percentage = (valid_count / 33) * 100
            frame_valid = "Yes" if valid_count >= 27 else "No"

            # Neck angle — side camera only
            if camera_role == "side":
                neck_angle = self.calculate_neck_angle(
                    smoothed_landmarks, facing_direction, frame_width, frame_height
                )
                neck_reba_score = self.calculate_neck_reba_score(neck_angle)
            else:
                neck_angle = float("nan")
                neck_reba_score = float("nan")

            # Trunk angle — side camera only
            if camera_role == "side":
                trunk_angle = self.calculate_trunk_angle(
                    smoothed_landmarks, facing_direction, frame_width, frame_height
                )
                trunk_reba_score = self.calculate_trunk_reba_score(
                    trunk_angle, self.trunk_baseline_angle0
                )
            else:
                trunk_angle = float("nan")
                trunk_reba_score = float("nan")

            # Leg raised and knee flexion — side camera only
            if camera_role == "side":
                leg_raised = self.calculate_leg_raised(
                    smoothed_landmarks,
                    frame_width,
                    frame_height,
                    self.leg_raised_baseline_mean0,
                )
                left_knee_angle, right_knee_angle = self.calculate_knee_flexion_angle(
                    smoothed_landmarks, frame_width, frame_height
                )
                leg_reba_score = self.calculate_leg_reba_score(
                    leg_raised, left_knee_angle, right_knee_angle
                )
            else:
                leg_raised = None
                left_knee_angle = float("nan")
                right_knee_angle = float("nan")
                leg_reba_score = float("nan")

            # Upper arm — side camera only
            if camera_role == "side":
                left_upper_arm_angle, right_upper_arm_angle = (
                    self.calculate_upper_arm_angle(
                        smoothed_landmarks, frame_width, frame_height
                    )
                )
                left_upper_arm_score = self.calculate_upper_arm_reba_score(
                    left_upper_arm_angle
                )
                right_upper_arm_score = self.calculate_upper_arm_reba_score(
                    right_upper_arm_angle
                )
                upper_arm_reba_score = max(left_upper_arm_score, right_upper_arm_score)
            else:
                left_upper_arm_angle = float("nan")
                right_upper_arm_angle = float("nan")
                upper_arm_reba_score = float("nan")

            # Lower arm — side camera only
            if camera_role == "side":
                left_lower_arm_angle, right_lower_arm_angle = (
                    self.calculate_lower_arm_angle(
                        smoothed_landmarks, frame_width, frame_height
                    )
                )
                left_lower_arm_score = self.calculate_lower_arm_reba_score(
                    left_lower_arm_angle
                )
                right_lower_arm_score = self.calculate_lower_arm_reba_score(
                    right_lower_arm_angle
                )
                lower_arm_reba_score = max(left_lower_arm_score, right_lower_arm_score)
            else:
                left_lower_arm_angle = float("nan")
                right_lower_arm_angle = float("nan")
                lower_arm_reba_score = float("nan")

            # Wrist flexion/extension — side camera only
            if camera_role == "side":
                left_wrist_angle, right_wrist_angle = self.calculate_wrist_angle(
                    smoothed_landmarks, facing_direction, frame_width, frame_height
                )
                left_wrist_score = self.calculate_wrist_reba_score(left_wrist_angle)
                right_wrist_score = self.calculate_wrist_reba_score(right_wrist_angle)
                wrist_reba_score = max(left_wrist_score, right_wrist_score)
            else:
                left_wrist_angle = float("nan")
                right_wrist_angle = float("nan")
                wrist_reba_score = float("nan")

            return {
                "detected": True,
                "landmarks": smoothed_landmarks,
                "raw_results": results,
                "valid_percentage": valid_percentage,
                "frame_valid": frame_valid,
                "stability": stability_scores,
                "new_ema": smoothed_landmarks,
                "new_prev": smoothed_landmarks,
                "neck_angle": neck_angle,
                "neck_reba_score": neck_reba_score,
                "trunk_angle": trunk_angle,
                "trunk_reba_score": trunk_reba_score,
                "leg_raised": leg_raised,
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "leg_reba_score": leg_reba_score,
                "left_upper_arm_angle": left_upper_arm_angle,
                "right_upper_arm_angle": right_upper_arm_angle,
                "upper_arm_reba_score": upper_arm_reba_score,
                "left_lower_arm_angle": left_lower_arm_angle,
                "right_lower_arm_angle": right_lower_arm_angle,
                "lower_arm_reba_score": lower_arm_reba_score,
                "left_wrist_angle": left_wrist_angle,
                "right_wrist_angle": right_wrist_angle,
                "wrist_reba_score": wrist_reba_score,
            }
        else:
            self.reset_stability(stability_history)
            return {
                "detected": False,
                "landmarks": None,
                "raw_results": None,
                "valid_percentage": float("nan"),
                "frame_valid": "No",
                "stability": [float("nan")] * 33,
                "new_ema": None,
                "new_prev": None,
                "neck_angle": float("nan"),
                "neck_reba_score": float("nan"),
                "neck_twisted": None,
                "neck_side_bent": None,
                "trunk_angle": float("nan"),
                "trunk_reba_score": float("nan"),
                "trunk_twisted": None,
                "trunk_side_bent": None,
                "leg_raised": None,
                "left_knee_angle": float("nan"),
                "right_knee_angle": float("nan"),
                "leg_reba_score": float("nan"),
                "left_upper_arm_angle": float("nan"),
                "right_upper_arm_angle": float("nan"),
                "upper_arm_reba_score": float("nan"),
                "shoulder_raised": None,
                "upper_arm_abducted": None,
                "left_lower_arm_angle": float("nan"),
                "right_lower_arm_angle": float("nan"),
                "lower_arm_reba_score": float("nan"),
                "left_wrist_angle": float("nan"),
                "right_wrist_angle": float("nan"),
                "wrist_reba_score": float("nan"),
                "left_wrist_bent": None,
                "right_wrist_bent": None,
                "wrist_bent_midline_score": float("nan"),
                "left_wrist_twisted": None,
                "right_wrist_twisted": None,
            }

    def mediapipe_processing_thread(self):
        """Separate thread for MediaPipe processing"""
        print("MediaPipe processing thread started")

        while self.running or not self.queue0.empty() or not self.queue1.empty():
            try:
                try:
                    frame0_data = self.queue0.get(timeout=0.1)
                    frame1_data = self.queue1.get(timeout=0.1)
                except queue.Empty:
                    if not self.running:
                        break
                    continue

                frame0, ts0, frame_num, was_dup0 = frame0_data
                frame1, ts1, frame_num, was_dup1 = frame1_data

                rotated_frame0 = self.rotate_frame_90cw(frame0)
                rotated_frame1 = self.rotate_frame_90cw(frame1)

                mp_result0 = self.process_mediapipe_frame(
                    rotated_frame0,
                    self.pose0,
                    self.ema_landmarks0,
                    self.prev_landmarks0,
                    self.stability_history0,
                    facing_direction=1,
                    frame_width=self.rotated_width0,
                    frame_height=self.rotated_height0,
                    camera_role="side",
                )
                mp_result1 = self.process_mediapipe_frame(
                    rotated_frame1,
                    self.pose1,
                    self.ema_landmarks1,
                    self.prev_landmarks1,
                    self.stability_history1,
                    facing_direction=-1,
                    frame_width=self.rotated_width1,
                    frame_height=self.rotated_height1,
                    camera_role="angled",
                )

                self.ema_landmarks0 = mp_result0["new_ema"]
                self.ema_landmarks1 = mp_result1["new_ema"]
                self.prev_landmarks0 = mp_result0["new_prev"]
                self.prev_landmarks1 = mp_result1["new_prev"]

                self.processed_queue0.put(
                    {
                        "frame": rotated_frame0,
                        "timestamp": ts0,
                        "frame_num": frame_num,
                        "was_duplicated": was_dup0,
                        "mp_result": mp_result0,
                    }
                )

                self.processed_queue1.put(
                    {
                        "frame": rotated_frame1,
                        "timestamp": ts1,
                        "frame_num": frame_num,
                        "was_duplicated": was_dup1,
                        "mp_result": mp_result1,
                    }
                )

            except Exception as e:
                print(f"MediaPipe processing error: {e}")
                continue

    def apply_kinematic_constraints(self, lms, trunk_len, w, h):
        corrected = copy.deepcopy(lms)
        # The strict outward sequence (Core -> Extremity)
        kinematic_chain = [
            (11, 13),
            (13, 15),
            (15, 19),
            (15, 17),
            (15, 21),
            (23, 25),
            (25, 27),
            (27, 31),
            (27, 29),
        ]
        for p1_idx, p2_idx in kinematic_chain:
            if (p1_idx, p2_idx) not in getattr(self, "max_limb_ratios", {}):
                continue
            max_dist = self.max_limb_ratios[(p1_idx, p2_idx)] * trunk_len

            x1, y1 = corrected[p1_idx]["x"] * w, corrected[p1_idx]["y"] * h
            x2, y2 = corrected[p2_idx]["x"] * w, corrected[p2_idx]["y"] * h

            dx, dy = x2 - x1, y2 - y1
            current_dist = np.hypot(dx, dy)

            if current_dist > max_dist:
                scale = max_dist / (current_dist + 1e-8)
                # Anchor logic: if p1 is real, move p2. If both projected, core (p1) anchors p2.
                if lms[p1_idx]["visibility"] > 0.5 or (
                    lms[p1_idx]["visibility"] <= 0.5
                    and lms[p2_idx]["visibility"] <= 0.5
                ):
                    corrected[p2_idx]["x"] = (x1 + dx * scale) / w
                    corrected[p2_idx]["y"] = (y1 + dy * scale) / h
                else:  # If p2 is real and p1 is projected
                    corrected[p1_idx]["x"] = (x2 - dx * scale) / w
                    corrected[p1_idx]["y"] = (y2 - dy * scale) / h
        return corrected

    def get_fused_landmarks(self, lms0, projected_data, weight, trunk_len, w, h):
        fused = copy.deepcopy(lms0)
        for i, (proj_x, proj_y) in projected_data.items():
            fused[i]["x"] = weight * lms0[i]["x"] + (1.0 - weight) * proj_x
            fused[i]["y"] = weight * lms0[i]["y"] + (1.0 - weight) * proj_y
        # Always apply constraints after fusion
        return self.apply_kinematic_constraints(fused, trunk_len, w, h)

    # CAMERA FUSION  -------------------------------------------------------------------------------------------------------

    def recalculate_angles_with_corrected_landmarks(
        self, corrected_landmarks, facing_direction, frame_width, frame_height
    ):
        """Recalculate all angle-based metrics using corrected landmarks"""

        """
        #Debug
        print(f"[Recalc] facing_direction: {facing_direction}")
        print(f"[Recalc] frame_width: {frame_width}, frame_height: {frame_height}")
        for i, lm in enumerate(corrected_landmarks):
            print(f"  [{i}] x={lm['x'] * frame_width:.1f}, y={lm['y'] * frame_height:.1f}, vis={lm['visibility']:.2f}")
        """

        neck_angle = self.calculate_neck_angle(
            corrected_landmarks, facing_direction, frame_width, frame_height
        )
        neck_reba_score = self.calculate_neck_reba_score(neck_angle)

        trunk_angle = self.calculate_trunk_angle(
            corrected_landmarks, facing_direction, frame_width, frame_height
        )
        trunk_reba_score = self.calculate_trunk_reba_score(
            trunk_angle, self.trunk_baseline_angle0
        )

        leg_raised = self.calculate_leg_raised(
            corrected_landmarks,
            frame_width,
            frame_height,
            self.leg_raised_baseline_mean0,
        )
        left_knee_angle, right_knee_angle = self.calculate_knee_flexion_angle(
            corrected_landmarks, frame_width, frame_height
        )
        leg_reba_score = self.calculate_leg_reba_score(
            leg_raised, left_knee_angle, right_knee_angle
        )

        left_upper_arm_angle, right_upper_arm_angle = self.calculate_upper_arm_angle(
            corrected_landmarks, frame_width, frame_height
        )
        left_upper_arm_score = self.calculate_upper_arm_reba_score(left_upper_arm_angle)
        right_upper_arm_score = self.calculate_upper_arm_reba_score(
            right_upper_arm_angle
        )
        upper_arm_reba_score = max(left_upper_arm_score, right_upper_arm_score)

        left_lower_arm_angle, right_lower_arm_angle = self.calculate_lower_arm_angle(
            corrected_landmarks, frame_width, frame_height
        )
        left_lower_arm_score = self.calculate_lower_arm_reba_score(left_lower_arm_angle)
        right_lower_arm_score = self.calculate_lower_arm_reba_score(
            right_lower_arm_angle
        )
        lower_arm_reba_score = max(left_lower_arm_score, right_lower_arm_score)

        left_wrist_angle, right_wrist_angle = self.calculate_wrist_angle(
            corrected_landmarks, facing_direction, frame_width, frame_height
        )
        left_wrist_score = self.calculate_wrist_reba_score(left_wrist_angle)
        right_wrist_score = self.calculate_wrist_reba_score(right_wrist_angle)
        wrist_reba_score = max(left_wrist_score, right_wrist_score)

        return {
            "neck_angle_corrected": neck_angle,
            "neck_reba_score_corrected": neck_reba_score,
            "trunk_angle_corrected": trunk_angle,
            "trunk_reba_score_corrected": trunk_reba_score,
            "leg_raised_corrected": leg_raised,
            "left_knee_angle_corrected": left_knee_angle,
            "right_knee_angle_corrected": right_knee_angle,
            "leg_reba_score_corrected": leg_reba_score,
            "left_upper_arm_angle_corrected": left_upper_arm_angle,
            "right_upper_arm_angle_corrected": right_upper_arm_angle,
            "upper_arm_reba_score_corrected": upper_arm_reba_score,
            "left_lower_arm_angle_corrected": left_lower_arm_angle,
            "right_lower_arm_angle_corrected": right_lower_arm_angle,
            "lower_arm_reba_score_corrected": lower_arm_reba_score,
            "left_wrist_angle_corrected": left_wrist_angle,
            "right_wrist_angle_corrected": right_wrist_angle,
            "wrist_reba_score_corrected": wrist_reba_score,
        }

    def calculate_trunk_length_2d(self, landmarks, frame_width, frame_height):
        """Calculate 2D trunk length in pixel space"""
        ls = np.array(
            [landmarks[11]["x"] * frame_width, landmarks[11]["y"] * frame_height]
        )
        rs = np.array(
            [landmarks[12]["x"] * frame_width, landmarks[12]["y"] * frame_height]
        )
        lh = np.array(
            [landmarks[23]["x"] * frame_width, landmarks[23]["y"] * frame_height]
        )
        rh = np.array(
            [landmarks[24]["x"] * frame_width, landmarks[24]["y"] * frame_height]
        )
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        return np.linalg.norm(shoulder_mid - hip_mid)

    def update_trunk_length(self, landmarks, frame_width, frame_height, camera_role):
        """Update last valid trunk length if shoulders and hips are all visible"""
        if all(landmarks[i]["visibility"] >= 0.5 for i in [11, 12, 23, 24]):
            trunk_len = self.calculate_trunk_length_2d(
                landmarks, frame_width, frame_height
            )
            if trunk_len > 0:
                if camera_role == "side":
                    self.last_valid_trunk_length0 = trunk_len
                else:
                    self.last_valid_trunk_length1 = trunk_len
            else:
                if camera_role == "side":
                    self.last_valid_trunk_length0 = self.trunk_length_baseline_mean0
                else:
                    self.last_valid_trunk_length1 = self.trunk_length_baseline_mean1

    def project_cam1_to_cam0(
        self,
        landmarks0,
        landmarks1,
        frame_width0,
        frame_height0,
        frame_width1,
        frame_height1,
    ):
        """
        Project low-visibility cam0 landmarks using cam1 landmarks.
        Normalizes by trunk length, applies -135deg rotation in XY plane, denormalizes to cam0 space.
        """

        corrected_landmarks = copy.deepcopy(landmarks0)
        corrected_indices = []

        # Update trunk lengths
        self.update_trunk_length(landmarks0, frame_width0, frame_height0, "side")
        self.update_trunk_length(landmarks1, frame_width1, frame_height1, "angled")

        trunk_len0 = self.last_valid_trunk_length0
        trunk_len1 = self.last_valid_trunk_length1

        if trunk_len0 is None or trunk_len1 is None:
            return copy.deepcopy(landmarks0), []

        # Cam0 hip midpoint
        hip_mid0_x = (
            landmarks0[23]["x"] * frame_width0 + landmarks0[24]["x"] * frame_width0
        ) / 2
        hip_mid0_y = (
            landmarks0[23]["y"] * frame_height0 + landmarks0[24]["y"] * frame_height0
        ) / 2

        # Cam1 hip midpoint
        hip_mid1_x = (
            landmarks1[23]["x"] * frame_width1 + landmarks1[24]["x"] * frame_width1
        ) / 2
        hip_mid1_y = (
            landmarks1[23]["y"] * frame_height1 + landmarks1[24]["y"] * frame_height1
        ) / 2

        # projected_storage = {}  # Store (x_norm, y_norm) for iterative tuning, comment out to turn off tuning

        for i in range(33):
            if landmarks0[i]["visibility"] < 0.5:
                # Cam1 landmark in pixel space relative to hip mid
                x1 = landmarks1[i]["x"] * frame_width1 - hip_mid1_x
                y1 = landmarks1[i]["y"] * frame_height1 - hip_mid1_y

                # Normalize by cam1 trunk length
                x1_norm = x1 / (trunk_len1 + 1e-8)
                y1_norm = y1 / (trunk_len1 + 1e-8)

                # Flip X to match cam0 coordinate direction
                x1_norm = -x1_norm

                # Denormalize using cam0 trunk length
                x0 = x1_norm * trunk_len0 + hip_mid0_x
                y0 = y1_norm * trunk_len0 + hip_mid0_y

                # --- CHANGE START HERE ---
                # 'offset_ratio' is the value you stored during calibration (pixels / baseline_trunk)
                offset_ratio = self.cam1_projection_x_offsets.get(i, 0)

                # We multiply the ratio by the CURRENT live trunk length (trunk_len0), This makes the offset "elastic" so it shrinks when you bend
                dynamic_x_offset = offset_ratio * trunk_len0

                # --- START FUSION PATCH ---
                w_base = 0.70

                # The projected coordinates from Cam 1 (normalized to Cam 0 space)
                proj_x_norm = (x0 + dynamic_x_offset) / frame_width0
                proj_y_norm = y0 / frame_height0

                # Store this for the weight tuning loop, comment this out to turn off tuning
                # projected_storage[i] = (proj_x_norm, proj_y_norm)

                # Original Cam 0 coordinates
                orig_x_norm = landmarks0[i]["x"]
                orig_y_norm = landmarks0[i]["y"]

                # Bias-heavy fusion formula: w * Cam0 + (1 - w) * ProjectedCam1
                corrected_landmarks[i]["x"] = (
                    w_base * orig_x_norm + (1.0 - w_base) * proj_x_norm
                )
                corrected_landmarks[i]["y"] = (
                    w_base * orig_y_norm + (1.0 - w_base) * proj_y_norm
                )
                # --- END FUSION PATCH ---
                corrected_indices.append(i)

        """
        # Debug
        print(f"[Proj] Corrected {len(corrected_indices)} landmarks: {corrected_indices}")
        for i in corrected_indices:
            print(
                f"  [{i}] cam0_orig=({landmarks0[i]['x'] * frame_width0:.1f},{landmarks0[i]['y'] * frame_height0:.1f}) "
                f"vis={landmarks0[i]['visibility']:.2f} → "
                f"corrected=({corrected_landmarks[i]['x'] * frame_width0:.1f},{corrected_landmarks[i]['y'] * frame_height0:.1f})")
        """

        """ uncomment this to turn off the weighted fusion tuning
        # --- WEIGHT TUNING & CSV EXPORT ---
        # Define joint groups for the NaN rule
        joint_sets = {
            'trunk': [11, 12, 23, 24],
            'neck': [7, 8, 11, 12, 23, 24],
            'r_up': [12, 14, 24],
            'r_lo': [12, 14, 16],
            'legs': [23, 24, 25, 26, 27, 28]
        }

        tuning_row = []
        weights = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]

        # We need a facing_direction (assuming +1 for Cam0, adjust if you have a variable)
        fd = getattr(self, 'current_facing_direction', 1)

        for w in weights:
            # 1. Create a set of landmarks for this specific weight
            temp_lms = self.get_fused_landmarks(landmarks0, projected_storage, w, trunk_len0, frame_width0,
                                                frame_height0)

            # 2. Calculate Angles
            t_ang = self.calculate_trunk_angle(temp_lms, fd, frame_width0, frame_height0)
            n_ang = self.calculate_neck_angle(temp_lms, fd, frame_width0, frame_height0)
            _, r_up_ang = self.calculate_upper_arm_angle(temp_lms, frame_width0, frame_height0)
            _, r_lo_ang = self.calculate_lower_arm_angle(temp_lms, frame_width0, frame_height0)
            l_knee, r_knee = self.calculate_knee_flexion_angle(temp_lms, frame_width0, frame_height0)
            leg_max = max(l_knee, r_knee)

            # 3. Apply NaN Rule: Only record if at least one joint was a projected one
            angles = [t_ang, n_ang, r_up_ang, r_lo_ang, leg_max]
            keys = ['trunk', 'neck', 'r_up', 'r_lo', 'legs']

            for ang, key in zip(angles, keys):
                if any(idx in corrected_indices for idx in joint_sets[key]):
                    tuning_row.append(ang)
                else:
                    tuning_row.append(float('nan'))

        # Save to separate CSV
        with open(self.tuning_csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow(tuning_row)
        # """  # uncomment this to turn off the weighted fusion tuning

        # Apply constraints to your default 0.75 corrected_landmarks for the main pipeline return
        final_landmarks = self.apply_kinematic_constraints(
            corrected_landmarks, trunk_len0, frame_width0, frame_height0
        )

        return final_landmarks, corrected_indices

    ######################################################## DISPLAY #######################################################

    def timer_capture_frames(self):
        """Timer thread that captures frames from both cameras at exact intervals"""
        interval = 1.0 / self.synchronized_fps

        print("Waiting for initial frames from both cameras...")
        while self.running:
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()

            if ret0 and ret1:
                self.last_frame0 = frame0.copy()
                self.last_frame1 = frame1.copy()
                print("Initial frames captured successfully!")
                break
            time.sleep(0.01)

        next_capture_time = time.time()

        while self.running:
            current_time = time.time()

            if current_time >= next_capture_time:
                timestamp = time.time()

                ret0, frame0 = self.cap0.read()
                ret1, frame1 = self.cap1.read()

                if ret0:
                    self.last_frame0 = frame0.copy()
                    frame_to_use0 = frame0
                    was_duplicated0 = False
                else:
                    frame_to_use0 = self.last_frame0
                    self.duplicated_frames0 += 1
                    was_duplicated0 = True

                if ret1:
                    self.last_frame1 = frame1.copy()
                    frame_to_use1 = frame1
                    was_duplicated1 = False
                else:
                    frame_to_use1 = self.last_frame1
                    self.duplicated_frames1 += 1
                    was_duplicated1 = True

                self.frame_timestamps0.append(timestamp)
                self.frame_timestamps1.append(timestamp)

                frame_number = self.total_frames_written + 1

                try:
                    self.queue0.put_nowait(
                        (frame_to_use0.copy(), timestamp, frame_number, was_duplicated0)
                    )
                except queue.Full:
                    self.dropped_frames0 += 1

                try:
                    self.queue1.put_nowait(
                        (frame_to_use1.copy(), timestamp, frame_number, was_duplicated1)
                    )
                except queue.Full:
                    self.dropped_frames1 += 1

                self.total_frames_written += 1
                next_capture_time += interval

            else:
                sleep_time = min(0.001, (next_capture_time - current_time) / 2)
                time.sleep(sleep_time)

    def add_timestamp(self, frame, timestamp):
        """Add timestamp overlay to frame"""
        dt = datetime.fromtimestamp(timestamp)
        timestamp_text = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(
            frame,
            timestamp_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame

    def draw_landmarks_on_frame(self, frame, mp_result, draw_midpoints=True):
        """Draw MediaPipe landmarks on frame for display only"""
        if mp_result["detected"] and mp_result["raw_results"]:
            self.mp_drawing.draw_landmarks(
                frame,
                mp_result["raw_results"].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # Only draw custom midpoints if requested
            if draw_midpoints:
                h, w = frame.shape[:2]
                lms = mp_result["raw_results"].pose_landmarks.landmark
                yellow = (0, 255, 255)
                black = (0, 0, 0)

                # Define all pairs: Shoulder(11,12), Hip(23,24), L-Hand(17,19), R-Hand(18,20)
                midpoint_pairs = [(11, 12), (23, 24), (17, 19), (18, 20)]

                for p1, p2 in midpoint_pairs:
                    m_x = int(((lms[p1].x + lms[p2].x) / 2) * w)
                    m_y = int(((lms[p1].y + lms[p2].y) / 2) * h)
                    cv2.circle(frame, (m_x, m_y), 6, yellow, -1)
                    cv2.circle(frame, (m_x, m_y), 8, black, 2)
        return frame

    def draw_corrected_landmarks(
        self, frame, corrected_landmarks, corrected_indices, frame_width, frame_height
    ):
        """Draw corrected landmarks in red on cam0 frame"""
        # 1. Draw individual corrected landmarks in red
        for idx in corrected_indices:
            x = int(corrected_landmarks[idx]["x"] * frame_width)
            y = int(corrected_landmarks[idx]["y"] * frame_height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # 2. Calculate Midpoints
        midpoint_pairs = [
            (11, 12),  # Shoulder
            (23, 24),  # Hip
            (17, 19),  # Left Finger Mid
            (18, 20),  # Right Finger Mid
        ]

        for p1, p2 in midpoint_pairs:
            mid_x = int(
                (corrected_landmarks[p1]["x"] + corrected_landmarks[p2]["x"])
                / 2
                * frame_width
            )
            mid_y = int(
                (corrected_landmarks[p1]["y"] + corrected_landmarks[p2]["y"])
                / 2
                * frame_height
            )

            # Draw styled midpoint (Yellow with Black outline)
            cv2.circle(frame, (mid_x, mid_y), 6, (0, 255, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 8, (0, 0, 0), 2)

        return frame

    def draw_info_panel(
        self, combined_frame, mp_result0, mp_result1, corrected_results=None
    ):
        """Draw white info panel below combined frame with angle metrics"""
        panel_width = combined_frame.shape[1]
        num_cols = 3
        col_w = panel_width // num_cols

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.8
        thick = 2
        black = (0, 0, 0)
        line_h = 48

        def val(v, unit="deg"):
            return f"{v:.1f}{unit}" if v is not None and not np.isnan(v) else "N/A"

        def bval(v):
            return "Yes" if v is True else ("No" if v is False else "N/A")

        def fetch(mp_result, mp_key, corr_dict=None, corr_key=None):
            if corr_dict is not None and corr_key and corr_key in corr_dict:
                return corr_dict[corr_key]
            return mp_result.get(mp_key, float("nan"))

        # Build metrics into 3 columns
        cols = [
            [
                f"Neck: {val(fetch(mp_result0, 'neck_angle', corrected_results, 'neck_angle_corrected'))}",
                f"Trunk: {val(fetch(mp_result0, 'trunk_angle', corrected_results, 'trunk_angle_corrected'))}",
                f"Leg Raised: {bval(mp_result0.get('leg_raised'))}",
                f"L Knee: {val(fetch(mp_result0, 'left_knee_angle', corrected_results, 'left_knee_angle_corrected'))}",
                f"R Knee: {val(fetch(mp_result0, 'right_knee_angle', corrected_results, 'right_knee_angle_corrected'))}",
            ],
            [
                f"L Upper Arm: {val(fetch(mp_result0, 'left_upper_arm_angle', corrected_results, 'left_upper_arm_angle_corrected'))}",
                f"R Upper Arm: {val(fetch(mp_result0, 'right_upper_arm_angle', corrected_results, 'right_upper_arm_angle_corrected'))}",
                f"L Lower Arm: {val(fetch(mp_result0, 'left_lower_arm_angle', corrected_results, 'left_lower_arm_angle_corrected'))}",
                f"R Lower Arm: {val(fetch(mp_result0, 'right_lower_arm_angle', corrected_results, 'right_lower_arm_angle_corrected'))}",
            ],
            [
                f"L Wrist: {val(fetch(mp_result0, 'left_wrist_angle', corrected_results, 'left_wrist_angle_corrected'))}",
                f"R Wrist: {val(fetch(mp_result0, 'right_wrist_angle', corrected_results, 'right_wrist_angle_corrected'))}",
            ],
        ]

        # Calculate required panel height
        max_lines = max(len(c) for c in cols)
        header_h = 50
        panel_height = header_h + max_lines * line_h + 20
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255

        # Header
        title = "ANGLES - CORRECTED" if corrected_results else "ANGLES"
        cv2.putText(panel, title, (20, 35), font, 1.0, black, thick)
        cv2.line(
            panel, (0, header_h - 5), (panel_width, header_h - 5), (150, 150, 150), 1
        )

        # Column dividers
        for c in range(1, num_cols):
            cv2.line(
                panel, (c * col_w, 0), (c * col_w, panel_height), (200, 200, 200), 1
            )

        # Draw text
        for col_idx, lines in enumerate(cols):
            x = col_idx * col_w + 15
            y = header_h + line_h - 10
            for line in lines:
                cv2.putText(panel, line, (x, y), font, fs, black, thick)
                y += line_h

        return np.vstack((combined_frame, panel))

    def letterbox_image(self, image, window_width, window_height):
        """Resizes image to fit window while maintaining aspect ratio with black padding."""
        img_h, img_w = image.shape[:2]

        # Calculate the scale to fit the image inside the window
        scale = min(window_width / img_w, window_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize the image with the calculated scale
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a black canvas of the window size
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Center the resized image on the canvas
        x_offset = (window_width - new_w) // 2
        y_offset = (window_height - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def show_adjustment_form(self):
        """Show tkinter form for all post-recording adjustments"""

        result = {}
        root = tk.Tk()
        root.geometry("600x800")  # Set a default starting size (Width x Height)
        root.minsize(500, 600)  # Optional: Prevent the user from making it too small
        root.title("REBA Post-Recording Adjustments")
        root.resizable(True, True)

        # Main frame with scrollbar
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # --- CUSTOM SCROLLBAR ---
        def custom_yview(*args):
            canvas.yview(*args)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=custom_yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        pad = {"padx": 10, "pady": 3}

        # --- LOGO SETUP (Header) ---
        try:
            logo = Image.open("USM_logo.png")

            # Use thumbnail to preserve the aspect ratio (prevents stretching)
            # You can adjust the 200, 200 to be larger or smaller if needed!
            logo.thumbnail((200, 200), Image.Resampling.LANCZOS)

            logo_img = ImageTk.PhotoImage(logo)

            # Create a label to hold the image INSIDE the scrollable frame
            logo_label = tk.Label(scrollable_frame, image=logo_img)
            logo_label.image = logo_img  # Prevent garbage collection
            logo_label.pack(pady=(15, 10))  # Pack it at the very top of the form
        except Exception as e:
            print(f"Error loading logo: {e}")

        # Define your specific hex colors for purple and orange here
        header_colors = itertools.cycle(["#5f208c", "#f89a3b"])

        def section_header(parent, text):
            current_bg_color = next(header_colors)
            tk.Label(
                parent,
                text=text,
                font=("Arial", 12, "bold"),
                bg=current_bg_color,
                fg="white",
                anchor="w",
                padx=10,
            ).pack(fill="x", pady=(10, 2))

        def checkbox_row(parent, text, var):
            tk.Checkbutton(parent, text=text, variable=var, font=("Arial", 10)).pack(
                anchor="w", **pad
            )

        # Variables
        # Neck
        neck_twisted_var = tk.BooleanVar()
        neck_side_bent_var = tk.BooleanVar()

        # Trunk
        trunk_twisted_var = tk.BooleanVar()
        trunk_side_bent_var = tk.BooleanVar()

        # Force/Load
        weight_var = tk.StringVar(value="0")
        shock_var = tk.BooleanVar()

        # Upper arm
        shoulder_raised_var = tk.BooleanVar()
        upper_arm_abducted_var = tk.BooleanVar()
        arm_supported_var = tk.BooleanVar()
        leaning_var = tk.BooleanVar()

        # Wrist
        left_wrist_bent_var = tk.BooleanVar()
        right_wrist_bent_var = tk.BooleanVar()
        left_wrist_twisted_var = tk.BooleanVar()
        right_wrist_twisted_var = tk.BooleanVar()

        # Coupling
        coupling_var = tk.IntVar(value=0)

        # Activity
        activity_static_var = tk.BooleanVar()
        activity_repeated_var = tk.BooleanVar()
        activity_rapid_var = tk.BooleanVar()

        # ── NECK ──
        section_header(scrollable_frame, "NECK")
        checkbox_row(scrollable_frame, "Neck Twisted", neck_twisted_var)
        checkbox_row(scrollable_frame, "Neck Side Bent", neck_side_bent_var)

        # ── TRUNK ──
        section_header(scrollable_frame, "TRUNK")
        checkbox_row(scrollable_frame, "Trunk Twisted", trunk_twisted_var)
        checkbox_row(scrollable_frame, "Trunk Side Bent", trunk_side_bent_var)

        # ── FORCE / LOAD ──
        section_header(scrollable_frame, "FORCE / LOAD")
        weight_frame = tk.Frame(scrollable_frame)
        weight_frame.pack(anchor="w", **pad)
        tk.Label(
            weight_frame, text="Weight of object held (kg):", font=("Arial", 10)
        ).pack(side="left")
        tk.Entry(
            weight_frame, textvariable=weight_var, width=8, font=("Arial", 10)
        ).pack(side="left", padx=5)
        checkbox_row(scrollable_frame, "Sudden / Shock Loading", shock_var)

        # ── UPPER ARM ──
        section_header(scrollable_frame, "UPPER ARM")
        checkbox_row(scrollable_frame, "Shoulder Raised", shoulder_raised_var)
        checkbox_row(scrollable_frame, "Upper Arm Abducted", upper_arm_abducted_var)
        checkbox_row(scrollable_frame, "Arm Supported", arm_supported_var)
        checkbox_row(scrollable_frame, "Leaning", leaning_var)

        # ── WRIST ──
        section_header(scrollable_frame, "WRIST")
        checkbox_row(
            scrollable_frame, "Left Wrist Bent from Midline", left_wrist_bent_var
        )
        checkbox_row(
            scrollable_frame, "Right Wrist Bent from Midline", right_wrist_bent_var
        )
        checkbox_row(scrollable_frame, "Left Wrist Twisted", left_wrist_twisted_var)
        checkbox_row(scrollable_frame, "Right Wrist Twisted", right_wrist_twisted_var)

        # ── COUPLING ──
        section_header(scrollable_frame, "COUPLING")
        coupling_options = [
            (0, "Good — Well fitting handle and mid range power grip"),
            (
                1,
                "Fair — Acceptable but not ideal hand hold or coupling acceptable with another body part",
            ),
            (2, "Poor — Hand hold not acceptable but possible"),
            (3, "Unacceptable — No handles, awkward, unsafe with any body part"),
        ]
        for val, desc in coupling_options:
            tk.Radiobutton(
                scrollable_frame,
                text=desc,
                variable=coupling_var,
                value=val,
                font=("Arial", 10),
                wraplength=500,
                justify="left",
            ).pack(anchor="w", **pad)

        # ── ACTIVITY ──
        section_header(scrollable_frame, "ACTIVITY")
        checkbox_row(
            scrollable_frame,
            "One or more body parts static, held > 1 minute",
            activity_static_var,
        )
        checkbox_row(
            scrollable_frame,
            "Repeated small range actions, > 4 times per minute",
            activity_repeated_var,
        )
        checkbox_row(
            scrollable_frame,
            "Rapid large range changes in posture or unstable base",
            activity_rapid_var,
        )

        # ── SUBMIT ──
        def on_submit():
            try:
                weight = float(weight_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Input", "Please enter a valid number for weight."
                )
                return

            # Force/load score
            if weight < 5:
                force_score = 0
            elif weight <= 10:
                force_score = 1
            else:
                force_score = 2
            if shock_var.get():
                force_score += 1

            # Arm supported score
            arm_supported_score = (
                -1 if (arm_supported_var.get() or leaning_var.get()) else 0
            )

            # Activity score
            activity_score = sum(
                [
                    activity_static_var.get(),
                    activity_repeated_var.get(),
                    activity_rapid_var.get(),
                ]
            )

            result["neck_twisted"] = neck_twisted_var.get()
            result["neck_side_bent"] = neck_side_bent_var.get()
            result["trunk_twisted"] = trunk_twisted_var.get()
            result["trunk_side_bent"] = trunk_side_bent_var.get()
            result["force_load_score"] = force_score
            result["shoulder_raised"] = shoulder_raised_var.get()
            result["upper_arm_abducted"] = upper_arm_abducted_var.get()
            result["arm_supported_score"] = arm_supported_score
            result["left_wrist_bent"] = left_wrist_bent_var.get()
            result["right_wrist_bent"] = right_wrist_bent_var.get()
            result["left_wrist_twisted"] = left_wrist_twisted_var.get()
            result["right_wrist_twisted"] = right_wrist_twisted_var.get()
            result["coupling_score"] = coupling_var.get()
            result["activity_score"] = activity_score

            root.destroy()

        tk.Button(
            scrollable_frame,
            text="Submit",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=8,
            command=on_submit,
        ).pack(pady=20)

        def _on_mousewheel(event):
            # For Windows/macOS
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind to the canvas so it scrolls when the mouse is over it
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # For Linux compatibility
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        root.mainloop()
        return result

    ###################################################### CALIBRATION #####################################################
    def calibrate_projection_offsets(
        self,
        landmarks0,
        landmarks1,
        frame_width0,
        frame_height0,
        frame_width1,
        frame_height1,
    ):
        """
        Compute per-landmark x offsets for cam1 → cam0 projection during calibration.
        Subject must face 90° with cam0 on right side.
        """
        import math

        self.update_trunk_length(landmarks0, frame_width0, frame_height0, "side")
        self.update_trunk_length(landmarks1, frame_width1, frame_height1, "angled")

        trunk_len0 = self.last_valid_trunk_length0
        trunk_len1 = self.last_valid_trunk_length1

        if trunk_len0 is None or trunk_len1 is None:
            return

        hip_mid0_x = (
            landmarks0[23]["x"] * frame_width0 + landmarks0[24]["x"] * frame_width0
        ) / 2
        hip_mid0_y = (
            landmarks0[23]["y"] * frame_height0 + landmarks0[24]["y"] * frame_height0
        ) / 2
        hip_mid1_x = (
            landmarks1[23]["x"] * frame_width1 + landmarks1[24]["x"] * frame_width1
        ) / 2
        hip_mid1_y = (
            landmarks1[23]["y"] * frame_height1 + landmarks1[24]["y"] * frame_height1
        ) / 2

        angle = math.radians(0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        def project_landmark(idx):
            """Project a single landmark from cam1 to cam0 space, return projected x"""
            x1 = landmarks1[idx]["x"] * frame_width1 - hip_mid1_x
            y1 = landmarks1[idx]["y"] * frame_height1 - hip_mid1_y
            x1_norm = -(x1 / (trunk_len1 + 1e-8))
            y1_norm = y1 / (trunk_len1 + 1e-8)
            x_rot = x1_norm * cos_a - y1_norm * sin_a
            y_rot = x1_norm * sin_a + y1_norm * cos_a
            x0 = x_rot * trunk_len0 + hip_mid0_x
            return x0

        offsets = {}

        for left_idx, right_idx in self.PAIRED_LANDMARKS.items():
            # Right side offset
            right_cam0_x = landmarks0[right_idx]["x"] * frame_width0
            projected_right_x = project_landmark(right_idx)
            offsets[right_idx] = right_cam0_x - projected_right_x

            # Left side offset
            projected_left_x = project_landmark(left_idx)
            if landmarks0[left_idx]["visibility"] >= 0.5:
                left_cam0_x = landmarks0[left_idx]["x"] * frame_width0
            else:
                # Use right side as substitute
                left_cam0_x = right_cam0_x
            offsets[left_idx] = left_cam0_x - projected_left_x

        self.cam1_projection_x_offsets = offsets

        """
        # Debug
        for idx, offset in sorted(offsets.items()):
            print(f"  [{idx}] {self.landmark_names[idx]}: x_offset={offset:.2f}px")
        """

    def measure_calib_limbs(self, lms0, w0, h0):
        # Calculate current trunk length for ratio scaling
        shoulder_mid_y = (lms0[11]["y"] + lms0[12]["y"]) / 2 * h0
        hip_mid_y = (lms0[23]["y"] + lms0[24]["y"]) / 2 * h0
        current_trunk = abs(hip_mid_y - shoulder_mid_y)
        if current_trunk == 0:
            return

        # Map: Left Pair (to constrain) -> Right Pair (to measure)
        pairs_map = {
            (11, 13): (12, 14),
            (13, 15): (14, 16),
            (15, 19): (16, 20),
            (15, 17): (16, 18),
            (15, 21): (16, 22),
            (23, 25): (24, 26),
            (25, 27): (26, 28),
            (27, 31): (28, 32),
            (27, 29): (28, 30),
        }

        for left_pair, right_pair in pairs_map.items():
            p1 = lms0[right_pair[0]]
            p2 = lms0[right_pair[1]]
            # Only measure if both right-side joints are highly visible
            if p1["visibility"] > 0.5 and p2["visibility"] > 0.5:
                dx = (p1["x"] - p2["x"]) * w0
                dy = (p1["y"] - p2["y"]) * h0
                dist = np.hypot(dx, dy)
                self.calib_limb_lengths[left_pair].append(dist / current_trunk)

    def calibrate_fps(self):
        """Calibrate and measure real FPS from both cameras"""
        print("\n" + "=" * 50)
        print("CALIBRATING CAMERAS - Please wait 5 seconds...")
        print("=" * 50)

        frame0_rotated = None
        frame1_rotated = None

        calibration_duration = 5.0
        frame_count0 = 0
        frame_count1 = 0

        # trunk
        trunk_angles0 = []
        trunk_angles1 = []

        # leg
        leg_raised_diffs0 = []
        leg_raised_diffs1 = []

        neck_angle_diffs0 = []
        left_knee_angle_diffs0 = []
        right_knee_angle_diffs0 = []
        left_upper_arm_angle_diffs0 = []
        right_upper_arm_angle_diffs0 = []
        left_lower_arm_angle_diffs0 = []
        right_lower_arm_angle_diffs0 = []
        left_wrist_angle_diffs0 = []
        right_wrist_angle_diffs0 = []

        # Camera Fusion
        trunk_length_diffs0 = []
        trunk_length_diffs1 = []

        cv2.namedWindow("Calibration - REBA Detection System", cv2.WINDOW_NORMAL)

        start_time = time.time()

        projection_offset_frames = {}
        result0 = None
        result1 = None
        lms0 = None
        lms1 = None

        while True:
            elapsed = time.time() - start_time

            if elapsed >= calibration_duration:
                break

            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()

            if ret0:
                frame_count0 += 1
            if ret1:
                frame_count1 += 1

            # t0 = time.time() # Debug for latency

            if ret0:
                frame0_rotated = self.rotate_frame_90cw(frame0)
                rgb0 = cv2.cvtColor(frame0_rotated, cv2.COLOR_BGR2RGB)
                result0 = self.pose0.process(rgb0)
                if result0.pose_landmarks:
                    lms0 = [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                        for lm in result0.pose_landmarks.landmark
                    ]
                    trunk_angle0 = self.calculate_trunk_angle(
                        lms0, 1, self.rotated_width0, self.rotated_height0
                    )
                    trunk_angles0.append(trunk_angle0)
                    leg_raised_diff0 = self.calculate_ankles_y_diff(
                        lms0, self.rotated_width0, self.rotated_height0
                    )
                    leg_raised_diffs0.append(leg_raised_diff0)
                    neck_angle0 = self.calculate_neck_angle(
                        lms0, 1, self.rotated_width0, self.rotated_height0
                    )
                    neck_angle_diffs0.append(neck_angle0)

                    left_knee0, right_knee0 = self.calculate_knee_flexion_angle(
                        lms0, self.rotated_width0, self.rotated_height0
                    )
                    left_knee_angle_diffs0.append(left_knee0)
                    right_knee_angle_diffs0.append(right_knee0)

                    left_ua0, right_ua0 = self.calculate_upper_arm_angle(
                        lms0, self.rotated_width0, self.rotated_height0
                    )
                    left_upper_arm_angle_diffs0.append(left_ua0)
                    right_upper_arm_angle_diffs0.append(right_ua0)

                    left_la0, right_la0 = self.calculate_lower_arm_angle(
                        lms0, self.rotated_width0, self.rotated_height0
                    )
                    left_lower_arm_angle_diffs0.append(left_la0)
                    right_lower_arm_angle_diffs0.append(right_la0)

                    left_wrist0, right_wrist0 = self.calculate_wrist_angle(
                        lms0, 1, self.rotated_width0, self.rotated_height0
                    )
                    left_wrist_angle_diffs0.append(left_wrist0)
                    right_wrist_angle_diffs0.append(right_wrist0)
                    # Camera Fusion
                    # 2D trunk length cam0
                    ls0 = np.array(
                        [
                            lms0[11]["x"] * self.rotated_width0,
                            lms0[11]["y"] * self.rotated_height0,
                        ]
                    )
                    rs0 = np.array(
                        [
                            lms0[12]["x"] * self.rotated_width0,
                            lms0[12]["y"] * self.rotated_height0,
                        ]
                    )
                    lh0 = np.array(
                        [
                            lms0[23]["x"] * self.rotated_width0,
                            lms0[23]["y"] * self.rotated_height0,
                        ]
                    )
                    rh0 = np.array(
                        [
                            lms0[24]["x"] * self.rotated_width0,
                            lms0[24]["y"] * self.rotated_height0,
                        ]
                    )
                    shoulder_mid0 = (ls0 + rs0) / 2
                    hip_mid0 = (lh0 + rh0) / 2
                    trunk_len0 = np.linalg.norm(shoulder_mid0 - hip_mid0)
                    if trunk_len0 > 0:
                        trunk_length_diffs0.append(trunk_len0)

            # t1 = time.time()  # Debug for latency

            if ret1:
                frame1_rotated = self.rotate_frame_90cw(frame1)
                rgb1 = cv2.cvtColor(frame1_rotated, cv2.COLOR_BGR2RGB)
                result1 = self.pose1.process(rgb1)
                if result1.pose_landmarks:
                    lms1 = [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                        for lm in result1.pose_landmarks.landmark
                    ]
                    leg_raised_diff1 = self.calculate_ankles_y_diff(
                        lms1, self.rotated_width1, self.rotated_height1
                    )
                    leg_raised_diffs1.append(leg_raised_diff1)
                    # Camera fusion
                    # 2D trunk length cam1
                    ls1 = np.array(
                        [
                            lms1[11]["x"] * self.rotated_width1,
                            lms1[11]["y"] * self.rotated_height1,
                        ]
                    )
                    rs1 = np.array(
                        [
                            lms1[12]["x"] * self.rotated_width1,
                            lms1[12]["y"] * self.rotated_height1,
                        ]
                    )
                    lh1 = np.array(
                        [
                            lms1[23]["x"] * self.rotated_width1,
                            lms1[23]["y"] * self.rotated_height1,
                        ]
                    )
                    rh1 = np.array(
                        [
                            lms1[24]["x"] * self.rotated_width1,
                            lms1[24]["y"] * self.rotated_height1,
                        ]
                    )
                    shoulder_mid1 = (ls1 + rs1) / 2
                    hip_mid1 = (lh1 + rh1) / 2
                    trunk_len1 = np.linalg.norm(shoulder_mid1 - hip_mid1)
                    if trunk_len1 > 0:
                        trunk_length_diffs1.append(trunk_len1)

            # t2 = time.time()  # Debug for latency

            if ret0 and ret1:
                if result0 is not None and result1 is not None:
                    if result0.pose_landmarks and result1.pose_landmarks:
                        self.calibrate_projection_offsets(
                            lms0,
                            lms1,
                            self.rotated_width0,
                            self.rotated_height0,
                            self.rotated_width1,
                            self.rotated_height1,
                        )
                        self.measure_calib_limbs(
                            lms0, self.rotated_width0, self.rotated_height0
                        )

                    for idx, offset in self.cam1_projection_x_offsets.items():
                        if idx not in projection_offset_frames:
                            projection_offset_frames[idx] = []
                        projection_offset_frames[idx].append(offset)

            # t3 = time.time()  # Debug for latency

            if ret0 and ret1:
                frame0_rot = frame0_rotated
                frame1_rot = frame1_rotated

                scale = 0.5
                frame0_small = cv2.resize(
                    frame0_rot,
                    (
                        int(self.rotated_width0 * scale),
                        int(self.rotated_height0 * scale),
                    ),
                )
                frame1_small = cv2.resize(
                    frame1_rot,
                    (
                        int(self.rotated_width1 * scale),
                        int(self.rotated_height1 * scale),
                    ),
                )

                max_height = max(frame0_small.shape[0], frame1_small.shape[0])
                if frame0_small.shape[0] != max_height:
                    aspect = frame0_small.shape[1] / frame0_small.shape[0]
                    frame0_small = cv2.resize(
                        frame0_small, (int(max_height * aspect), max_height)
                    )
                if frame1_small.shape[0] != max_height:
                    aspect = frame1_small.shape[1] / frame1_small.shape[0]
                    frame1_small = cv2.resize(
                        frame1_small, (int(max_height * aspect), max_height)
                    )

                combined = np.hstack((frame0_small, frame1_small))

                # t3a = time.time()  # Debug for latency

                remaining = calibration_duration - elapsed
                calib_text = f"CALIBRATING... {remaining:.1f}s remaining"

                text_size = cv2.getTextSize(
                    calib_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
                )[0]
                box_x1 = (combined.shape[1] - text_size[0]) // 2 - 20
                box_y1 = combined.shape[0] // 2 - 60
                box_x2 = (combined.shape[1] + text_size[0]) // 2 + 20
                box_y2 = combined.shape[0] // 2 + 20
                cv2.rectangle(
                    combined, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1
                )
                cv2.putText(
                    combined,
                    calib_text,
                    ((combined.shape[1] - text_size[0]) // 2, combined.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3,
                )

                # t3b = time.time()  # Debug for latency

                cv2.imshow("Calibration - REBA Detection System", combined)
                cv2.waitKey(1)

                # t4 = time.time()  # Debug for latency
                # print(f"[Timing] calc&fuse0={1000 * (t1 - t0):.1f}ms calc&fuse1={1000 * (t2 - t1):.1f}ms projoffset={1000 * (t3 - t2):.1f}ms hstack={1000 * (t3a - t3):.1f}ms putext={1000 * (t3b - t3a):.1f}ms display={1000 * (t4 - t3b):.1f}ms")

        # trunk
        self.trunk_baseline_angle0 = np.mean(trunk_angles0) if trunk_angles0 else 0.0
        self.trunk_baseline_angle1 = np.mean(trunk_angles1) if trunk_angles1 else 0.0
        print(f"Trunk baseline angle cam0: {self.trunk_baseline_angle0:.4f}")
        print(f"Trunk baseline angle cam1: {self.trunk_baseline_angle1:.4f}")

        # legs
        self.leg_raised_baseline_mean0 = (
            np.mean(leg_raised_diffs0) if leg_raised_diffs0 else 0.1
        )
        self.leg_raised_baseline_mean1 = (
            np.mean(leg_raised_diffs1) if leg_raised_diffs1 else 0.1
        )
        self.leg_raised_baseline_mean1 = (
            np.mean(leg_raised_diffs1) if leg_raised_diffs1 else 0.1
        )
        print(f"Leg raised baseline cam0: {self.leg_raised_baseline_mean0:.4f}")
        print(f"Leg raised baseline cam1: {self.leg_raised_baseline_mean1:.4f}")

        self.neck_baseline_angle0 = (
            np.mean(neck_angle_diffs0) if neck_angle_diffs0 else 0.0
        )
        self.left_knee_baseline_angle0 = (
            np.mean(left_knee_angle_diffs0) if left_knee_angle_diffs0 else 0.0
        )
        self.right_knee_baseline_angle0 = (
            np.mean(right_knee_angle_diffs0) if right_knee_angle_diffs0 else 0.0
        )
        self.left_upper_arm_baseline_angle0 = (
            np.mean(left_upper_arm_angle_diffs0) if left_upper_arm_angle_diffs0 else 0.0
        )
        self.right_upper_arm_baseline_angle0 = (
            np.mean(right_upper_arm_angle_diffs0)
            if right_upper_arm_angle_diffs0
            else 0.0
        )
        self.left_lower_arm_baseline_angle0 = (
            np.mean(left_lower_arm_angle_diffs0) if left_lower_arm_angle_diffs0 else 0.0
        )
        self.right_lower_arm_baseline_angle0 = (
            np.mean(right_lower_arm_angle_diffs0)
            if right_lower_arm_angle_diffs0
            else 0.0
        )
        self.left_wrist_baseline_angle0 = (
            np.mean(left_wrist_angle_diffs0) if left_wrist_angle_diffs0 else 0.0
        )
        self.right_wrist_baseline_angle0 = (
            np.mean(right_wrist_angle_diffs0) if right_wrist_angle_diffs0 else 0.0
        )
        print(f"Neck baseline angle cam0: {self.neck_baseline_angle0:.4f}")
        print(f"Left knee baseline angle cam0: {self.left_knee_baseline_angle0:.4f}")
        print(f"Right knee baseline angle cam0: {self.right_knee_baseline_angle0:.4f}")
        print(
            f"Left upper arm baseline angle cam0: {self.left_upper_arm_baseline_angle0:.4f}"
        )
        print(
            f"Right upper arm baseline angle cam0: {self.right_upper_arm_baseline_angle0:.4f}"
        )
        print(
            f"Left lower arm baseline angle cam0: {self.left_lower_arm_baseline_angle0:.4f}"
        )
        print(
            f"Right lower arm baseline angle cam0: {self.right_lower_arm_baseline_angle0:.4f}"
        )
        print(f"Left wrist baseline angle cam0: {self.left_wrist_baseline_angle0:.4f}")
        print(
            f"Right wrist baseline angle cam0: {self.right_wrist_baseline_angle0:.4f}"
        )

        # camera fusion
        self.trunk_length_baseline_mean0 = (
            np.mean(trunk_length_diffs0) if trunk_length_diffs0 else 0.1
        )
        self.trunk_length_baseline_mean1 = (
            np.mean(trunk_length_diffs1) if trunk_length_diffs1 else 0.1
        )
        print(f"Trunk length baseline cam0: {self.trunk_length_baseline_mean0:.4f}")
        print(f"Trunk length baseline cam1: {self.trunk_length_baseline_mean1:.4f}")
        # Use the baseline trunk length of Cam 0 as the reference "scale"
        denominator = (
            self.trunk_length_baseline_mean0
            if self.trunk_length_baseline_mean0 > 0
            else 1.0
        )

        # Keep the name 'self.cam1_projection_x_offsets' but store the RATIO
        self.cam1_projection_x_offsets = {
            idx: np.mean(offsets) / denominator
            for idx, offsets in projection_offset_frames.items()
        }

        print(f"\n[Calibration] Final averaged projection Ratios:")
        for idx, ratio in sorted(self.cam1_projection_x_offsets.items()):
            name = (
                self.landmark_names[idx]
                if hasattr(self, "landmark_names")
                else f"landmark_{idx}"
            )
            print(
                f"  [{idx}] {name}: ratio={ratio:.4f}"
            )  # Now printing as a decimal ratio

        print("\n[Calibration] Final Max Limb Length Ratios:")
        for pair, ratios in self.calib_limb_lengths.items():
            if len(ratios) > 0:
                self.max_limb_ratios[pair] = np.mean(ratios)
                print(f"  {pair}: {self.max_limb_ratios[pair]:.4f}x trunk")

        # fps
        self.measured_fps0 = frame_count0 / calibration_duration
        self.measured_fps1 = frame_count1 / calibration_duration
        self.synchronized_fps = min(self.measured_fps0, self.measured_fps1)

        cv2.destroyWindow("Calibration - REBA Detection System")

        print(f"\nCalibration complete!")
        print(f"Camera 0 measured real FPS: {self.measured_fps0:.2f}")
        print(f"Camera 1 measured real FPS: {self.measured_fps1:.2f}")
        print(f"Using synchronized FPS: {self.synchronized_fps:.2f}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out0 = cv2.VideoWriter(
            self.temp_filename0,
            fourcc,
            self.synchronized_fps,
            (self.rotated_width0, self.rotated_height0),
        )
        self.out1 = cv2.VideoWriter(
            self.temp_filename1,
            fourcc,
            self.synchronized_fps,
            (self.rotated_width1, self.rotated_height1),
        )

        print(f"\nRecording will be saved as:")
        print(f"  - {self.filename0} @ true FPS (calculated after recording)")
        print(f"  - {self.filename1} @ true FPS (calculated after recording)")

    '''
    def calibrate_trunk_twisted(self):
        """Calibrate neck side bend at 3 facing angles: 0deg, 90deg, 135deg"""
        angles = ['0 degrees (directly toward Cam 0)', '90 degrees (facing front)', '135 degrees (toward Cam 1)']
        calibration_duration = 5.0

        cv2.namedWindow('Neck Side Bend Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Neck Side Bend Calibration', 1280, 720)

        for idx, angle_label in enumerate(angles):
            # Prompt to press C
            print(f"\nNeck side bend calibration round {idx + 1}/3: Please face {angle_label}")
            print("Press 'C' to start this round...")

            while True:
                ret0, frame0 = self.cap0.read()
                ret1, frame1 = self.cap1.read()

                if ret0 and ret1:
                    frame0_rot = self.rotate_frame_90cw(frame0)
                    frame1_rot = self.rotate_frame_90cw(frame1)

                    max_height = max(self.rotated_height0, self.rotated_height1)
                    if self.rotated_height0 != max_height:
                        aspect = self.rotated_width0 / self.rotated_height0
                        frame0_rot = cv2.resize(frame0_rot, (int(max_height * aspect), max_height))
                    if self.rotated_height1 != max_height:
                        aspect = self.rotated_width1 / self.rotated_height1
                        frame1_rot = cv2.resize(frame1_rot, (int(max_height * aspect), max_height))

                    combined = np.hstack((frame0_rot, frame1_rot))
                    prompt_text = f"Round {idx + 1}/3: Face {angle_label}"
                    prompt_text2 = "Press 'C' to start..."

                    overlay = combined.copy()
                    cv2.rectangle(overlay, (0, combined.shape[0] // 2 - 80),
                                  (combined.shape[1], combined.shape[0] // 2 + 60), (0, 0, 0), -1)
                    combined = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)

                    text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.putText(combined, prompt_text,
                                ((combined.shape[1] - text_size[0]) // 2, combined.shape[0] // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    text_size2 = cv2.getTextSize(prompt_text2, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.putText(combined, prompt_text2,
                                ((combined.shape[1] - text_size2[0]) // 2, combined.shape[0] // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    cv2.imshow('Neck Side Bend Calibration', combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break

            # Countdown 2 seconds
            countdown_start = time.time()
            while time.time() - countdown_start < 2.0:
                ret0, frame0 = self.cap0.read()
                ret1, frame1 = self.cap1.read()

                if ret0 and ret1:
                    frame0_rot = self.rotate_frame_90cw(frame0)
                    frame1_rot = self.rotate_frame_90cw(frame1)

                    max_height = max(self.rotated_height0, self.rotated_height1)
                    if self.rotated_height0 != max_height:
                        aspect = self.rotated_width0 / self.rotated_height0
                        frame0_rot = cv2.resize(frame0_rot, (int(max_height * aspect), max_height))
                    if self.rotated_height1 != max_height:
                        aspect = self.rotated_width1 / self.rotated_height1
                        frame1_rot = cv2.resize(frame1_rot, (int(max_height * aspect), max_height))

                    combined = np.hstack((frame0_rot, frame1_rot))
                    remaining = 2.0 - (time.time() - countdown_start)
                    countdown_text = f"Starting in {remaining:.1f}s..."

                    overlay = combined.copy()
                    cv2.rectangle(overlay, (0, combined.shape[0] // 2 - 60),
                                  (combined.shape[1], combined.shape[0] // 2 + 40), (0, 0, 0), -1)
                    combined = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)

                    text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    cv2.putText(combined, countdown_text,
                                ((combined.shape[1] - text_size[0]) // 2, combined.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                    cv2.imshow('Neck Side Bend Calibration', combined)
                    cv2.waitKey(1)

            # Calibration round
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed >= calibration_duration:
                    break

                ret0, frame0 = self.cap0.read()
                ret1, frame1 = self.cap1.read()

                if ret0:
                    frame0_rotated = self.rotate_frame_90cw(frame0)
                    rgb0 = cv2.cvtColor(frame0_rotated, cv2.COLOR_BGR2RGB)
                    result0 = self.pose0.process(rgb0)
                    if result0.pose_landmarks:
                        lms0 = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                                for lm in result0.pose_landmarks.landmark]

                if ret1:
                    frame1_rotated = self.rotate_frame_90cw(frame1)
                    rgb1 = cv2.cvtColor(frame1_rotated, cv2.COLOR_BGR2RGB)
                    result1 = self.pose1.process(rgb1)
                    if result1.pose_landmarks:
                        lms1 = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                                for lm in result1.pose_landmarks.landmark]

                if ret0 and ret1:
                    frame0_rot = self.rotate_frame_90cw(frame0)
                    frame1_rot = self.rotate_frame_90cw(frame1)

                    max_height = max(self.rotated_height0, self.rotated_height1)
                    if self.rotated_height0 != max_height:
                        aspect = self.rotated_width0 / self.rotated_height0
                        frame0_rot = cv2.resize(frame0_rot, (int(max_height * aspect), max_height))
                    if self.rotated_height1 != max_height:
                        aspect = self.rotated_width1 / self.rotated_height1
                        frame1_rot = cv2.resize(frame1_rot, (int(max_height * aspect), max_height))

                    combined = np.hstack((frame0_rot, frame1_rot))
                    remaining = calibration_duration - elapsed
                    calib_text = f"Round {idx + 1}/3: Face {angle_label} - {remaining:.1f}s remaining"

                    overlay = combined.copy()
                    cv2.rectangle(overlay, (0, combined.shape[0] // 2 - 60),
                                  (combined.shape[1], combined.shape[0] // 2 + 40), (0, 0, 0), -1)
                    combined = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)

                    text_size = cv2.getTextSize(calib_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.putText(combined, calib_text,
                                ((combined.shape[1] - text_size[0]) // 2, combined.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                    cv2.imshow('Neck Side Bend Calibration', combined)
                    cv2.waitKey(1)

            # Store calibration point (**perhaps can add in trunk twist later here)

            # After last round, just proceed
            if idx == len(angles) - 1:
                break

        cv2.destroyWindow('Neck Side Bend Calibration')
        print("\nNeck side bend calibration complete!")
    '''

    ######################################################### MAIN ########################################################

    def run(self):
        """Main function to run dual camera capture"""
        self.calibrate_fps()

        input("\nPress Enter to start recording...")
        print("Starting in 2 seconds...")
        time.sleep(2)

        self.running = True
        self.recording = True

        cv2.namedWindow(
            "Dual Camera - REBA Detection System",
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )

        timer_thread = threading.Thread(target=self.timer_capture_frames, daemon=True)
        timer_thread.start()

        mediapipe_thread = threading.Thread(
            target=self.mediapipe_processing_thread, daemon=True
        )
        mediapipe_thread.start()

        print("\nDual camera capture started with MediaPipe pose detection!")
        print(f"Capturing frames at exactly {self.synchronized_fps:.2f} FPS intervals")
        print("Press 'q' to stop recording and exit")
        print("-" * 50)

        frame_latencies = []
        try:
            while (
                self.running
                or not self.processed_queue0.empty()
                or not self.processed_queue1.empty()
            ):
                while not self.processed_queue0.empty():
                    processed0 = (
                        self.processed_queue0.get()
                    )  # processed0 is just a temporary variable
                    frame_num = processed0["frame_num"]
                    self.buffer0[frame_num] = processed0

                while not self.processed_queue1.empty():
                    processed1 = self.processed_queue1.get()
                    frame_num = processed1["frame_num"]
                    self.buffer1[frame_num] = processed1

                matching_frames = set(self.buffer0.keys()) & set(self.buffer1.keys())

                for frame_num in sorted(matching_frames):
                    processed0 = self.buffer0.pop(frame_num)
                    processed1 = self.buffer1.pop(frame_num)

                    frame0 = processed0["frame"]
                    frame1 = processed1["frame"]
                    ts0 = processed0["timestamp"]
                    ts1 = processed1["timestamp"]
                    mp_result0 = processed0["mp_result"]
                    mp_result1 = processed1["mp_result"]

                    timestamp_str = datetime.fromtimestamp(ts0).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )[:-3]

                    # t0 = time.time() # Debug for latency

                    # Rotation-based projection correction
                    corrected_results = None
                    corrected_indices = []
                    corrected_landmarks = None
                    if mp_result0["detected"] and mp_result1["detected"]:
                        corrected_landmarks, corrected_indices = (
                            self.project_cam1_to_cam0(
                                mp_result0["landmarks"],
                                mp_result1["landmarks"],
                                self.rotated_width0,
                                self.rotated_height0,
                                self.rotated_width1,
                                self.rotated_height1,
                            )
                        )
                        # t1 = time.time()  # Debug for latency
                        if corrected_indices:
                            corrected_results = (
                                self.recalculate_angles_with_corrected_landmarks(
                                    corrected_landmarks,
                                    facing_direction=1,
                                    frame_width=self.rotated_width0,
                                    frame_height=self.rotated_height0,
                                )
                            )
                            # t2 = time.time()  # Debug for latency

                    natural_results = self.calculate_natural_angles(
                        corrected_results, mp_result0
                    )
                    # t3 = time.time()  # Debug for latency

                    csv_row0 = [
                        processed0["frame_num"],
                        timestamp_str,
                        "Yes" if processed0["was_duplicated"] else "No",
                        mp_result0["valid_percentage"],
                        mp_result0["frame_valid"],
                        mp_result0["neck_angle"],
                        mp_result0["neck_reba_score"],
                        (
                            corrected_results["neck_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["neck_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["neck_angle_natural"],
                        natural_results["neck_reba_score_natural"],
                        mp_result0["trunk_angle"],
                        mp_result0["trunk_reba_score"],
                        (
                            corrected_results["trunk_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["trunk_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["trunk_angle_natural"],
                        natural_results["trunk_reba_score_natural"],
                        mp_result0["leg_raised"],
                        mp_result0["left_knee_angle"],
                        mp_result0["right_knee_angle"],
                        mp_result0["leg_reba_score"],
                        (
                            corrected_results["leg_raised_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["left_knee_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_knee_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["leg_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_knee_angle_natural"],
                        natural_results["right_knee_angle_natural"],
                        natural_results["leg_reba_score_natural"],
                        mp_result0["left_upper_arm_angle"],
                        mp_result0["right_upper_arm_angle"],
                        mp_result0["upper_arm_reba_score"],
                        (
                            corrected_results["left_upper_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_upper_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["upper_arm_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_upper_arm_angle_natural"],
                        natural_results["right_upper_arm_angle_natural"],
                        natural_results["upper_arm_reba_score_natural"],
                        mp_result0["left_lower_arm_angle"],
                        mp_result0["right_lower_arm_angle"],
                        mp_result0["lower_arm_reba_score"],
                        (
                            corrected_results["left_lower_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_lower_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["lower_arm_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_lower_arm_angle_natural"],
                        natural_results["right_lower_arm_angle_natural"],
                        natural_results["lower_arm_reba_score_natural"],
                        mp_result0["left_wrist_angle"],
                        mp_result0["right_wrist_angle"],
                        mp_result0["wrist_reba_score"],
                        (
                            corrected_results["left_wrist_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_wrist_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["wrist_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_wrist_angle_natural"],
                        natural_results["right_wrist_angle_natural"],
                        natural_results["wrist_reba_score_natural"],
                    ]

                    if mp_result0["detected"]:
                        for lm in mp_result0["landmarks"]:
                            csv_row0.append(lm["visibility"])
                    else:
                        csv_row0.extend([float("nan")] * 33)

                    csv_row0.extend(mp_result0["stability"])
                    self.frame_data0.append(csv_row0)

                    csv_row1 = [
                        processed0["frame_num"],
                        timestamp_str,
                        "Yes" if processed0["was_duplicated"] else "No",
                        mp_result1["valid_percentage"],
                        mp_result1["frame_valid"],
                        mp_result1["neck_angle"],
                        mp_result1["neck_reba_score"],
                        (
                            corrected_results["neck_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["neck_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["neck_angle_natural"],
                        natural_results["neck_reba_score_natural"],
                        mp_result1["trunk_angle"],
                        mp_result1["trunk_reba_score"],
                        (
                            corrected_results["trunk_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["trunk_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["trunk_angle_natural"],
                        natural_results["trunk_reba_score_natural"],
                        mp_result1["leg_raised"],
                        mp_result1["left_knee_angle"],
                        mp_result1["right_knee_angle"],
                        mp_result1["leg_reba_score"],
                        (
                            corrected_results["leg_raised_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["left_knee_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_knee_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["leg_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_knee_angle_natural"],
                        natural_results["right_knee_angle_natural"],
                        natural_results["leg_reba_score_natural"],
                        mp_result1["left_upper_arm_angle"],
                        mp_result1["right_upper_arm_angle"],
                        mp_result1["upper_arm_reba_score"],
                        (
                            corrected_results["left_upper_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_upper_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["upper_arm_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_upper_arm_angle_natural"],
                        natural_results["right_upper_arm_angle_natural"],
                        natural_results["upper_arm_reba_score_natural"],
                        mp_result1["left_lower_arm_angle"],
                        mp_result1["right_lower_arm_angle"],
                        mp_result1["lower_arm_reba_score"],
                        (
                            corrected_results["left_lower_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_lower_arm_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["lower_arm_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_lower_arm_angle_natural"],
                        natural_results["right_lower_arm_angle_natural"],
                        natural_results["lower_arm_reba_score_natural"],
                        mp_result1["left_wrist_angle"],
                        mp_result1["right_wrist_angle"],
                        mp_result1["wrist_reba_score"],
                        (
                            corrected_results["left_wrist_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["right_wrist_angle_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        (
                            corrected_results["wrist_reba_score_corrected"]
                            if corrected_results
                            else float("nan")
                        ),
                        natural_results["left_wrist_angle_natural"],
                        natural_results["right_wrist_angle_natural"],
                        natural_results["wrist_reba_score_natural"],
                    ]

                    if mp_result1["detected"]:
                        for lm in mp_result1["landmarks"]:
                            csv_row1.append(lm["visibility"])
                    else:
                        csv_row1.extend([float("nan")] * 33)

                    csv_row1.extend(mp_result1["stability"])
                    self.frame_data1.append(csv_row1)

                    # t4 = time.time()  # Debug for latency

                    frame0_display = self.add_timestamp(frame0.copy(), ts0)
                    frame1_display = self.add_timestamp(frame1.copy(), ts1)
                    frame_num_text = f"Frame: {processed0['frame_num']}"
                    cv2.putText(
                        frame0_display,
                        frame_num_text,
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        3,
                    )
                    cv2.putText(
                        frame1_display,
                        frame_num_text,
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        3,
                    )

                    # Check if correction is active for this frame
                    has_correction0 = (
                        corrected_results is not None and corrected_indices
                    )

                    # Draw raw skeleton; skip midpoints if correction exists
                    frame0_display = self.draw_landmarks_on_frame(
                        frame0_display, mp_result0, draw_midpoints=not has_correction0
                    )

                    # If correction exists, draw the red dots AND the corrected midpoints
                    if has_correction0:
                        frame0_display = self.draw_corrected_landmarks(
                            frame0_display,
                            corrected_landmarks,
                            corrected_indices,
                            self.rotated_width0,
                            self.rotated_height0,
                        )

                    # Cam 1 usually doesn't have corrections in your setup, so it keeps midpoints
                    frame1_display = self.draw_landmarks_on_frame(
                        frame1_display, mp_result1, draw_midpoints=True
                    )

                    # t4a = time.time()  # Debug for latency

                    scale = 0.5
                    frame0_small = cv2.resize(
                        frame0_display,
                        (
                            int(self.rotated_width0 * scale),
                            int(self.rotated_height0 * scale),
                        ),
                    )
                    frame1_small = cv2.resize(
                        frame1_display,
                        (
                            int(self.rotated_width1 * scale),
                            int(self.rotated_height1 * scale),
                        ),
                    )

                    max_height = max(frame0_small.shape[0], frame1_small.shape[0])
                    if frame0_small.shape[0] != max_height:
                        aspect = frame0_small.shape[1] / frame0_small.shape[0]
                        frame0_small = cv2.resize(
                            frame0_small, (int(max_height * aspect), max_height)
                        )
                    if frame1_small.shape[0] != max_height:
                        aspect = frame1_small.shape[1] / frame1_small.shape[0]
                        frame1_small = cv2.resize(
                            frame1_small, (int(max_height * aspect), max_height)
                        )

                    # t4b = time.time()  # Debug for latency

                    combined = np.hstack((frame0_small, frame1_small))
                    combined = self.draw_info_panel(
                        combined, mp_result0, mp_result1, corrected_results
                    )

                    # t5 = time.time()  # Debug for latency

                    latency = (time.time() - ts0) * 1000
                    frame_latencies.append(latency)

                    # 2. Get the current size of the window (how big you dragged it)
                    win_name = "Dual Camera - REBA Detection System"
                    win_rect = cv2.getWindowImageRect(win_name)

                    # win_rect returns [x, y, width, height]
                    if win_rect is not None and win_rect[2] > 0 and win_rect[3] > 0:
                        win_w, win_h = win_rect[2], win_rect[3]
                        # Apply the letterbox helper to keep aspect ratio perfect
                        final_display = self.letterbox_image(combined, win_w, win_h)
                        cv2.imshow(win_name, final_display)
                    else:
                        # If window isn't ready yet, just show the combined frame
                        cv2.imshow(win_name, combined)

                    # t6 = time.time()  # Debug for latency
                    # print(f"[Timing] projection={1000 * (t1 - t0):.1f}ms recalc={1000 * (t2 - t1):.1f}ms natural={1000 * (t3 - t2):.1f}ms csv={1000 * (t4 - t3):.1f}ms landmarks={1000 * (t4a - t4):.1f}ms framerotate={1000 * (t4b - t4a):.1f}ms panel={1000 * (t5 - t4b):.1f}ms display={1000 * (t6 - t5):.1f}ms")

                    if self.recording:
                        clean_frame0 = self.add_timestamp(frame0.copy(), ts0)
                        clean_frame1 = self.add_timestamp(frame1.copy(), ts1)
                        frame_num_text = f"Frame: {processed0['frame_num']}"
                        cv2.putText(
                            clean_frame0,
                            frame_num_text,
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            3,
                        )
                        cv2.putText(
                            clean_frame1,
                            frame_num_text,
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            3,
                        )
                        self.out0.write(clean_frame0)
                        self.out1.write(clean_frame1)

                if self.running:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        print("\nStopping capture... Draining remaining frames...")
                        self.running = False  # This stops the capture and signals workers to finish
                else:
                    # During draining, we still need waitKey to keep the window responsive
                    cv2.waitKey(1)

        finally:
            self.running = False
            time.sleep(0.2)

            remaining_frames = self.queue0.qsize() + self.queue1.qsize()
            if remaining_frames > 0:
                print(f"\nProcessing remaining {remaining_frames // 2} frame pairs...")

            while (
                not self.queue0.empty() or not self.queue1.empty()
            ) and mediapipe_thread.is_alive():
                queue0_size = self.queue0.qsize()
                queue1_size = self.queue1.qsize()

                if queue0_size > 0 or queue1_size > 0:
                    print(
                        f"  Remaining: Queue0={queue0_size}, Queue1={queue1_size}",
                        end="\r",
                    )

                time.sleep(0.05)

            print("\nAll frames captured and sent to MediaPipe!")

            print("Matching and saving remaining processed frames...")

            while not self.processed_queue0.empty():
                processed0 = self.processed_queue0.get()
                frame_num = processed0["frame_num"]
                self.buffer0[frame_num] = processed0

            while not self.processed_queue1.empty():
                processed1 = self.processed_queue1.get()
                frame_num = processed1["frame_num"]
                self.buffer1[frame_num] = processed1

            adjustment_results = self.show_adjustment_form()
            self.neck_twisted = adjustment_results["neck_twisted"]
            self.neck_side_bent = adjustment_results["neck_side_bent"]
            self.trunk_twisted = adjustment_results["trunk_twisted"]
            self.trunk_side_bent = adjustment_results["trunk_side_bent"]
            self.force_load_score = adjustment_results["force_load_score"]
            self.shoulder_raised = adjustment_results["shoulder_raised"]
            self.upper_arm_abducted = adjustment_results["upper_arm_abducted"]
            self.arm_supported_score = adjustment_results["arm_supported_score"]
            self.left_wrist_bent = adjustment_results["left_wrist_bent"]
            self.right_wrist_bent = adjustment_results["right_wrist_bent"]
            self.left_wrist_twisted = adjustment_results["left_wrist_twisted"]
            self.right_wrist_twisted = adjustment_results["right_wrist_twisted"]
            self.coupling_score = adjustment_results["coupling_score"]
            self.activity_score = adjustment_results["activity_score"]

            matching_frames = set(self.buffer0.keys()) & set(self.buffer1.keys())

            for frame_num in sorted(matching_frames):
                processed0 = self.buffer0.pop(frame_num)
                processed1 = self.buffer1.pop(frame_num)

                mp_result0 = processed0["mp_result"]
                mp_result1 = processed1["mp_result"]
                ts0 = processed0["timestamp"]

                timestamp_str = datetime.fromtimestamp(ts0).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3]

                # Rotation-based projection correction
                corrected_results = None
                corrected_indices = []
                corrected_landmarks = None
                if mp_result0["detected"] and mp_result1["detected"]:
                    corrected_landmarks, corrected_indices = self.project_cam1_to_cam0(
                        mp_result0["landmarks"],
                        mp_result1["landmarks"],
                        self.rotated_width0,
                        self.rotated_height0,
                        self.rotated_width1,
                        self.rotated_height1,
                    )
                    if corrected_indices:
                        corrected_results = (
                            self.recalculate_angles_with_corrected_landmarks(
                                corrected_landmarks,
                                facing_direction=1,
                                frame_width=self.rotated_width0,
                                frame_height=self.rotated_height0,
                            )
                        )

                natural_results = self.calculate_natural_angles(
                    corrected_results, mp_result0
                )

                csv_row0 = [
                    processed0["frame_num"],
                    timestamp_str,
                    "Yes" if processed0["was_duplicated"] else "No",
                    mp_result0["valid_percentage"],
                    mp_result0["frame_valid"],
                    mp_result0["neck_angle"],
                    mp_result0["neck_reba_score"],
                    (
                        corrected_results["neck_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["neck_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["neck_angle_natural"],
                    natural_results["neck_reba_score_natural"],
                    mp_result0["trunk_angle"],
                    mp_result0["trunk_reba_score"],
                    (
                        corrected_results["trunk_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["trunk_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["trunk_angle_natural"],
                    natural_results["trunk_reba_score_natural"],
                    mp_result0["leg_raised"],
                    mp_result0["left_knee_angle"],
                    mp_result0["right_knee_angle"],
                    mp_result0["leg_reba_score"],
                    (
                        corrected_results["leg_raised_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["left_knee_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_knee_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["leg_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_knee_angle_natural"],
                    natural_results["right_knee_angle_natural"],
                    natural_results["leg_reba_score_natural"],
                    mp_result0["left_upper_arm_angle"],
                    mp_result0["right_upper_arm_angle"],
                    mp_result0["upper_arm_reba_score"],
                    (
                        corrected_results["left_upper_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_upper_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["upper_arm_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_upper_arm_angle_natural"],
                    natural_results["right_upper_arm_angle_natural"],
                    natural_results["upper_arm_reba_score_natural"],
                    mp_result0["left_lower_arm_angle"],
                    mp_result0["right_lower_arm_angle"],
                    mp_result0["lower_arm_reba_score"],
                    (
                        corrected_results["left_lower_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_lower_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["lower_arm_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_lower_arm_angle_natural"],
                    natural_results["right_lower_arm_angle_natural"],
                    natural_results["lower_arm_reba_score_natural"],
                    mp_result0["left_wrist_angle"],
                    mp_result0["right_wrist_angle"],
                    mp_result0["wrist_reba_score"],
                    (
                        corrected_results["left_wrist_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_wrist_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["wrist_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_wrist_angle_natural"],
                    natural_results["right_wrist_angle_natural"],
                    natural_results["wrist_reba_score_natural"],
                ]

                if mp_result0["detected"]:
                    for lm in mp_result0["landmarks"]:
                        csv_row0.append(lm["visibility"])
                else:
                    csv_row0.extend([float("nan")] * 33)

                csv_row0.extend(mp_result0["stability"])
                self.frame_data0.append(csv_row0)

                csv_row1 = [
                    processed0["frame_num"],
                    timestamp_str,
                    "Yes" if processed0["was_duplicated"] else "No",
                    mp_result1["valid_percentage"],
                    mp_result1["frame_valid"],
                    mp_result1["neck_angle"],
                    mp_result1["neck_reba_score"],
                    (
                        corrected_results["neck_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["neck_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["neck_angle_natural"],
                    natural_results["neck_reba_score_natural"],
                    mp_result1["trunk_angle"],
                    mp_result1["trunk_reba_score"],
                    (
                        corrected_results["trunk_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["trunk_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["trunk_angle_natural"],
                    natural_results["trunk_reba_score_natural"],
                    mp_result1["leg_raised"],
                    mp_result1["left_knee_angle"],
                    mp_result1["right_knee_angle"],
                    mp_result1["leg_reba_score"],
                    (
                        corrected_results["leg_raised_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["left_knee_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_knee_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["leg_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_knee_angle_natural"],
                    natural_results["right_knee_angle_natural"],
                    natural_results["leg_reba_score_natural"],
                    mp_result1["left_upper_arm_angle"],
                    mp_result1["right_upper_arm_angle"],
                    mp_result1["upper_arm_reba_score"],
                    (
                        corrected_results["left_upper_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_upper_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["upper_arm_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_upper_arm_angle_natural"],
                    natural_results["right_upper_arm_angle_natural"],
                    natural_results["upper_arm_reba_score_natural"],
                    mp_result1["left_lower_arm_angle"],
                    mp_result1["right_lower_arm_angle"],
                    mp_result1["lower_arm_reba_score"],
                    (
                        corrected_results["left_lower_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_lower_arm_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["lower_arm_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_lower_arm_angle_natural"],
                    natural_results["right_lower_arm_angle_natural"],
                    natural_results["lower_arm_reba_score_natural"],
                    mp_result1["left_wrist_angle"],
                    mp_result1["right_wrist_angle"],
                    mp_result1["wrist_reba_score"],
                    (
                        corrected_results["left_wrist_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["right_wrist_angle_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    (
                        corrected_results["wrist_reba_score_corrected"]
                        if corrected_results
                        else float("nan")
                    ),
                    natural_results["left_wrist_angle_natural"],
                    natural_results["right_wrist_angle_natural"],
                    natural_results["wrist_reba_score_natural"],
                ]

                if mp_result1["detected"]:
                    for lm in mp_result1["landmarks"]:
                        csv_row1.append(lm["visibility"])
                else:
                    csv_row1.extend([float("nan")] * 33)

                csv_row1.extend(mp_result1["stability"])
                self.frame_data1.append(csv_row1)

                frame0 = processed0["frame"]
                frame1 = processed1["frame"]
                clean_frame0 = self.add_timestamp(frame0.copy(), ts0)
                clean_frame1 = self.add_timestamp(
                    frame1.copy(), processed1["timestamp"]
                )
                frame_num_text = f"Frame: {processed0['frame_num']}"
                cv2.putText(
                    clean_frame0,
                    frame_num_text,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3,
                )
                cv2.putText(
                    clean_frame1,
                    frame_num_text,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3,
                )
                self.out0.write(clean_frame0)
                self.out1.write(clean_frame1)

            # patching
            for row in self.frame_data0 + self.frame_data1:
                # --- 1. Neck ---
                neck_reba = row[6]
                neck_reba_natural = row[10]
                neck_adj = (1 if self.neck_twisted else 0) + (
                    1 if self.neck_side_bent else 0
                )

                final_neck = (
                    neck_reba + neck_adj if not np.isnan(neck_reba) else float("nan")
                )
                final_neck_natural = (
                    neck_reba_natural + neck_adj
                    if not np.isnan(neck_reba_natural)
                    else float("nan")
                )

                row.insert(11, self.neck_twisted)
                row.insert(12, self.neck_side_bent)
                row.insert(13, final_neck)
                row.insert(14, final_neck_natural)
                # Total Shift: +4

                # --- 2. Trunk ---
                trunk_reba = row[18] if not np.isnan(row[18]) else row[16]
                trunk_reba_natural = row[20]

                trunk_adj = (1 if self.trunk_twisted else 0) + (
                    1 if self.trunk_side_bent else 0
                )
                final_trunk = (
                    trunk_reba + trunk_adj if not np.isnan(trunk_reba) else float("nan")
                )
                final_trunk_natural = (
                    trunk_reba_natural + trunk_adj
                    if not np.isnan(trunk_reba_natural)
                    else float("nan")
                )

                row.insert(21, self.trunk_twisted)
                row.insert(22, self.trunk_side_bent)
                row.insert(23, final_trunk)
                row.insert(24, final_trunk_natural)

                # --- 3. Leg & Table A ---
                leg_reba = row[28]
                leg_reba_corrected = row[32]
                leg_reba_natural = row[35]

                final_leg = (
                    leg_reba_corrected if not np.isnan(leg_reba_corrected) else leg_reba
                )
                final_leg_natural = leg_reba_natural

                row.insert(36, final_leg)
                row.insert(37, final_leg_natural)

                table_a = (
                    self.calculate_reba_table_a(row[13], row[23], final_leg)
                    if not np.any(np.isnan([row[13], row[23], final_leg]))
                    else float("nan")
                )
                table_a_natural = (
                    self.calculate_reba_table_a(row[14], row[24], final_leg_natural)
                    if not np.any(np.isnan([row[14], row[24], final_leg_natural]))
                    else float("nan")
                )

                row.insert(38, table_a)
                row.insert(39, table_a_natural)
                row.insert(40, self.force_load_score)

                score_a = self.calculate_score_a(table_a, self.force_load_score)
                score_a_natural = (
                    self.calculate_score_a(table_a_natural, self.force_load_score)
                    if not np.isnan(table_a_natural)
                    else float("nan")
                )
                row.insert(41, score_a)
                row.insert(42, score_a_natural)
                # Total Shift: +15

                # --- 4. Upper Arm ---
                ua_reba = row[45]
                ua_reba_corrected = row[48]
                ua_reba_natural = row[51]
                ua_adj = (1 if self.shoulder_raised else 0) + (
                    1 if self.upper_arm_abducted else 0
                )

                final_ua = (
                    max(1, ua_reba_corrected + ua_adj + self.arm_supported_score)
                    if not np.isnan(ua_reba_corrected)
                    else (
                        max(1, ua_reba + ua_adj + self.arm_supported_score)
                        if not np.isnan(ua_reba)
                        else float("nan")
                    )
                )
                final_ua_natural = (
                    max(1, ua_reba_natural + ua_adj + self.arm_supported_score)
                    if not np.isnan(ua_reba_natural)
                    else float("nan")
                )

                row.insert(52, self.shoulder_raised)
                row.insert(53, self.upper_arm_abducted)
                row.insert(54, self.arm_supported_score)
                row.insert(55, final_ua)
                row.insert(56, final_ua_natural)

                # --- 5. Lower Arm ---
                la_reba = row[59]
                la_reba_corrected = row[62]
                la_reba_natural = row[65]
                la_base = (
                    la_reba_corrected if not np.isnan(la_reba_corrected) else la_reba
                )
                la_base_natural = la_reba_natural

                # --- 6. Wrist ---
                wrist_reba = row[68]
                wrist_reba_corrected = row[71]
                wrist_reba_natural = row[74]

                wrist_adj = (
                    1
                    if (
                        self.left_wrist_bent
                        or self.right_wrist_bent
                        or self.left_wrist_twisted
                        or self.right_wrist_twisted
                    )
                    else 0
                )
                wrist_base = (
                    wrist_reba_corrected
                    if not np.isnan(wrist_reba_corrected)
                    else wrist_reba
                )
                wrist_base_natural = wrist_reba_natural

                final_wrist = (
                    wrist_base + wrist_adj if not np.isnan(wrist_base) else float("nan")
                )
                final_wrist_natural = (
                    wrist_base_natural + wrist_adj
                    if not np.isnan(wrist_base_natural)
                    else float("nan")
                )

                row.insert(75, self.left_wrist_bent)
                row.insert(76, self.right_wrist_bent)
                row.insert(
                    77, 1 if (self.left_wrist_bent or self.right_wrist_bent) else 0
                )
                row.insert(78, self.left_wrist_twisted)
                row.insert(79, self.right_wrist_twisted)
                row.insert(80, wrist_adj)
                row.insert(81, final_wrist)
                row.insert(82, final_wrist_natural)
                # Total Shift: +28

                # --- 7. Table B, Coupling, Score B ---
                table_b = (
                    self.calculate_reba_table_b(la_base, row[55], final_wrist)
                    if not np.any(np.isnan([la_base, row[55], final_wrist]))
                    else float("nan")
                )
                table_b_natural = (
                    self.calculate_reba_table_b(
                        la_base_natural, row[56], final_wrist_natural
                    )
                    if not np.any(
                        np.isnan([la_base_natural, row[56], final_wrist_natural])
                    )
                    else float("nan")
                )

                row.insert(83, table_b)
                row.insert(84, table_b_natural)
                row.insert(85, self.coupling_score)

                score_b = self.calculate_score_b(table_b, self.coupling_score)
                score_b_natural = (
                    self.calculate_score_b(table_b_natural, self.coupling_score)
                    if not np.isnan(table_b_natural)
                    else float("nan")
                )
                row.insert(86, score_b)
                row.insert(87, score_b_natural)

                # --- 8. Table C & Final ---
                table_c = (
                    self.calculate_reba_table_c(row[41], row[86])
                    if not np.any(np.isnan([row[41], row[86]]))
                    else float("nan")
                )
                table_c_natural = (
                    self.calculate_reba_table_c(row[42], row[87])
                    if not np.any(np.isnan([row[42], row[87]]))
                    else float("nan")
                )

                row.insert(88, table_c)
                row.insert(89, table_c_natural)
                row.insert(90, self.activity_score)
                row.insert(
                    91,
                    (
                        (table_c + self.activity_score)
                        if not np.isnan(table_c)
                        else float("nan")
                    ),
                )
                row.insert(
                    92,
                    (
                        (table_c_natural + self.activity_score)
                        if not np.isnan(table_c_natural)
                        else float("nan")
                    ),
                )

                # --- 9. Action Levels ---
                def get_action_level(score):
                    if np.isnan(score):
                        return float("nan")
                    s = round(score)
                    if s <= 1:
                        return 0
                    elif s <= 3:
                        return 1
                    elif s <= 7:
                        return 2
                    elif s <= 10:
                        return 3
                    else:
                        return 4

                row.insert(93, get_action_level(row[91]))
                row.insert(94, get_action_level(row[92]))

                # --- START SINGLE-CAM CALCULATION PATCH ---
                # 1. Calculate Table A (Uncorrected)
                # Using neck (idx 6), trunk (idx 16), and leg (idx 28)
                table_a_singlecam = self.calculate_reba_table_a(
                    row[6], row[16], row[28]
                )
                row.insert(95, table_a_singlecam)
                score_a_singlecam = self.calculate_score_a(
                    table_a_singlecam, self.force_load_score
                )
                row.insert(96, score_a_singlecam)

                # 2. Calculate Table B (Uncorrected)
                # Using upper arm (idx 45), lower arm (idx 59), and wrist (idx 68)
                table_b_singlecam = self.calculate_reba_table_b(
                    row[59], row[45], row[68]
                )
                row.insert(97, table_b_singlecam)
                score_b_singlecam = self.calculate_score_b(
                    table_b_singlecam, self.coupling_score
                )
                row.insert(98, score_b_singlecam)

                # 3. Calculate Table C and Final REBA (Uncorrected)
                table_c_singlecam = self.calculate_reba_table_c(
                    score_a_singlecam, score_b_singlecam
                )
                row.insert(99, table_c_singlecam)
                reba_score_singlecam = (
                    table_c_singlecam + row[90]
                )  # table_c + activity_score
                row.insert(100, reba_score_singlecam)

                action_level_singlecam = get_action_level(reba_score_singlecam)
                row.insert(101, action_level_singlecam)
                # --- END SINGLE-CAM CALCULATION PATCH ---

            # latency patching
            for i, row in enumerate(self.frame_data0):
                row.append(
                    frame_latencies[i] if i < len(frame_latencies) else float("nan")
                )
            for i, row in enumerate(self.frame_data1):
                row.append(
                    frame_latencies[i] if i < len(frame_latencies) else float("nan")
                )

            if (
                self.buffer0 or self.buffer1
            ):  # checks any remaining frames that couldn't be matched
                print("\n" + "!" * 50)
                print("WARNING: Unpaired frames detected and discarded:")
                if self.buffer0:
                    unpaired0 = sorted(self.buffer0.keys())
                    print(f"  Camera 0 unpaired frames: {unpaired0}")
                if self.buffer1:
                    unpaired1 = sorted(self.buffer1.keys())
                    print(f"  Camera 1 unpaired frames: {unpaired1}")
                print("!" * 50)

            # --- 9. Summary Statistics & Final Display ---
            # Using only cam0 data as it contains the primary records
            if self.frame_data0:
                # Extract scores from indices 91 and 92, explicitly ignoring NaNs
                reba_list = [
                    row[91] for row in self.frame_data0 if not np.isnan(row[91])
                ]
                reba_nat_list = [
                    row[92] for row in self.frame_data0 if not np.isnan(row[92])
                ]

                # Calculate means (fallback to 0.0 to prevent division-by-zero crashes if list is empty)
                mean_reba = np.mean(reba_list) if reba_list else 0.0
                mean_reba_nat = np.mean(reba_nat_list) if reba_nat_list else 0.0

                # Helper function to determine Risk Level based on REBA score
                def get_reba_risk_text(score):
                    s = round(score)
                    if s <= 1:
                        return "Level 0: Negligible risk", (0, 255, 0)  # Green
                    elif s <= 3:
                        return "Level 1: Low risk, change may be needed", (
                            100,
                            255,
                            100,
                        )  # Light Green
                    elif s <= 7:
                        return "Level 2: Medium risk, further invest. needed", (
                            0,
                            255,
                            255,
                        )  # Yellow
                    elif s <= 10:
                        return "Level 3: High risk, implement change soon", (
                            0,
                            165,
                            255,
                        )  # Orange
                    else:
                        return "Level 4: Very high risk, change immediately", (
                            0,
                            0,
                            255,
                        )  # Red

                risk_text_reba, color_reba = get_reba_risk_text(mean_reba)
                risk_text_nat, color_nat = get_reba_risk_text(mean_reba_nat)

                # Create a larger clean, black summary window (400 height x 750 width) to fit the text
                summary_img = np.zeros((400, 750, 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Draw the text overlay
                cv2.putText(
                    summary_img,
                    "SESSION SUMMARY (CAM 0)",
                    (200, 40),
                    font,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                # Standard REBA
                cv2.putText(
                    summary_img,
                    f"Avg REBA Score: {mean_reba:.2f}",
                    (30, 100),
                    font,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    summary_img,
                    f"-> {risk_text_reba}",
                    (50, 130),
                    font,
                    0.6,
                    color_reba,
                    1,
                )

                # Natural REBA
                cv2.putText(
                    summary_img,
                    f"Avg REBA Score (Nat): {mean_reba_nat:.2f}",
                    (30, 190),
                    font,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    summary_img,
                    f"-> {risk_text_nat}",
                    (50, 220),
                    font,
                    0.6,
                    color_nat,
                    1,
                )

                # Footer stats
                cv2.putText(
                    summary_img,
                    f"Valid Frames Evaluated: {len(reba_list)}",
                    (30, 290),
                    font,
                    0.6,
                    (150, 150, 150),
                    1,
                )
                cv2.putText(
                    summary_img,
                    "Press 'X' to close and cleanup...",
                    (200, 350),
                    font,
                    0.6,
                    (0, 0, 255),
                    1,
                )

                # Show window
                cv2.imshow("Processing Complete", summary_img)

                # Trap the script in this loop until 'X' or 'x' is pressed
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("x") or key == ord("X"):
                        break

                # Clear the window from the screen once the key is pressed
                cv2.destroyWindow("Processing Complete")
            else:
                print("Notice: No valid frame data in Cam 0 to calculate means.")

            # --- Append Summary Stats to the Bottom of Cam0 Data ---
            # 1. Grab frame stats before we alter the list length
            total_processed_frames = len(self.frame_data0)

            # 2. Get dropped/dupe stats (Double-check these variable names match your class!)
            dropped_cam0 = getattr(self, "dropped_frames0", 0)
            dropped_cam1 = getattr(self, "dropped_frames1", 0)
            dupes_cam0 = getattr(self, "duplicated_frames0", 0)
            dupes_cam1 = getattr(self, "duplicated_frames1", 0)
            # (Assuming total captured is processed + dropped)
            total_captured = total_processed_frames + dropped_cam0

            dupe_rate0 = (
                (dupes_cam0 / total_captured * 100) if total_captured > 0 else 0.0
            )
            dupe_rate1 = (
                (dupes_cam1 / total_captured * 100) if total_captured > 0 else 0.0
            )
            drop_rate0 = (
                (dropped_cam0 / total_captured * 100) if total_captured > 0 else 0.0
            )
            drop_rate1 = (
                (dropped_cam1 / total_captured * 100) if total_captured > 0 else 0.0
            )

            # 3. Helper function to keep CSV rows uniform
            total_cols = len(self.frame_data0[0]) if self.frame_data0 else 100

            def create_padded_row(col1, col2=""):
                row = [col1, col2]
                row.extend([""] * (total_cols - len(row)))
                return row

            # 4. Append the data to self.frame_data0
            self.frame_data0.extend([create_padded_row("", "")] * 3)
            self.frame_data0.append(create_padded_row("--- RUN SUMMARY STATS ---", ""))

            # Baselines
            self.frame_data0.append(
                create_padded_row(
                    "Trunk baseline angle cam0", f"{self.trunk_baseline_angle0:.4f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Trunk baseline angle cam1", f"{self.trunk_baseline_angle1:.4f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Leg raised baseline cam0", f"{self.leg_raised_baseline_mean0:.4f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Leg raised baseline cam1", f"{self.leg_raised_baseline_mean1:.4f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Neck baseline angle cam0", f"{self.neck_baseline_angle0:.4f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Left knee baseline angle cam0",
                    f"{self.left_knee_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Right knee baseline angle cam0",
                    f"{self.right_knee_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Left upper arm baseline angle cam0",
                    f"{self.left_upper_arm_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Right upper arm baseline angle cam0",
                    f"{self.right_upper_arm_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Left lower arm baseline angle cam0",
                    f"{self.left_lower_arm_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Right lower arm baseline angle cam0",
                    f"{self.right_lower_arm_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Left wrist baseline angle cam0",
                    f"{self.left_wrist_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Right wrist baseline angle cam0",
                    f"{self.right_wrist_baseline_angle0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Trunk length baseline cam0",
                    f"{self.trunk_length_baseline_mean0:.4f}",
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Trunk length baseline cam1",
                    f"{self.trunk_length_baseline_mean1:.4f}",
                )
            )

            # Offsets
            self.frame_data0.append(create_padded_row("", ""))
            self.frame_data0.append(
                create_padded_row(
                    "[Calibration] Final averaged projection offsets:", ""
                )
            )
            for idx, offset in sorted(self.cam1_projection_x_offsets.items()):
                name = (
                    self.landmark_names[idx]
                    if hasattr(self, "landmark_names")
                    else f"landmark_{idx}"
                )
                self.frame_data0.append(
                    create_padded_row(f"[{idx}] {name}", f"x_offset={offset:.2f}px")
                )

            # FPS
            self.frame_data0.append(create_padded_row("", ""))
            self.frame_data0.append(create_padded_row("Calibration complete!", ""))
            self.frame_data0.append(
                create_padded_row(
                    "Camera 0 measured real FPS", f"{self.measured_fps0:.2f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Camera 1 measured real FPS", f"{self.measured_fps1:.2f}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Using synchronized FPS", f"{self.synchronized_fps:.2f}"
                )
            )

            # Frame Stats
            self.frame_data0.append(create_padded_row("", ""))
            self.frame_data0.append(
                create_padded_row("Total frames captured", f"{total_captured}")
            )
            self.frame_data0.append(
                create_padded_row(
                    "Camera 0 frames dropped (queue full)", f"{dropped_cam0}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Camera 1 frames dropped (queue full)", f"{dropped_cam1}"
                )
            )
            self.frame_data0.append(
                create_padded_row(
                    "Total frames processed in CSV", f"{total_processed_frames}"
                )
            )
            self.frame_data0.append(
                create_padded_row("Camera 0 duplicated frames", f"{dupes_cam0}")
            )
            self.frame_data0.append(
                create_padded_row("Camera 1 duplicated frames", f"{dupes_cam1}")
            )
            self.frame_data0.append(
                create_padded_row("Camera 0 duplication rate", f"{dupe_rate0:.2f}%")
            )
            self.frame_data0.append(
                create_padded_row("Camera 1 duplication rate", f"{dupe_rate1:.2f}%")
            )
            self.frame_data0.append(
                create_padded_row("Camera 0 drop rate", f"{drop_rate0:.2f}%")
            )
            self.frame_data0.append(
                create_padded_row("Camera 1 drop rate", f"{drop_rate1:.2f}%")
            )

            self.cleanup()

    def check_ffmpeg_available(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def reencode_with_ffmpeg(self, input_file, output_file, fps):
        """Re-encode video using FFmpeg"""
        try:
            command = [
                "ffmpeg",
                "-i",
                input_file,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-r",
                str(fps),
                "-y",
                output_file,
            ]

            result = subprocess.run(command, capture_output=True, timeout=300)

            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"FFmpeg encoding failed: {e}")
            return False

    def reencode_videos(self):
        """Re-encode videos with true FPS"""
        print("\nCalculating true FPS from frame timestamps...")

        if len(self.frame_timestamps0) > 1:
            actual_duration0 = self.frame_timestamps0[-1] - self.frame_timestamps0[0]
            true_fps0 = (
                len(self.frame_timestamps0) - 1
            ) / actual_duration0  # checked, this is correct
        else:
            true_fps0 = self.synchronized_fps

        if len(self.frame_timestamps1) > 1:
            actual_duration1 = self.frame_timestamps1[-1] - self.frame_timestamps1[0]
            true_fps1 = (len(self.frame_timestamps1) - 1) / actual_duration1
        else:
            true_fps1 = self.synchronized_fps

        self.true_fps = (true_fps0 + true_fps1) / 2

        print(f"Camera 0 true FPS: {true_fps0:.2f}")
        print(f"Camera 1 true FPS: {true_fps1:.2f}")
        print(f"Using average true FPS: {self.true_fps:.2f}")

        if self.check_ffmpeg_available():
            print("\nRe-encoding videos with FFmpeg...")

            success0 = self.reencode_with_ffmpeg(
                self.temp_filename0, self.filename0, self.true_fps
            )
            success1 = self.reencode_with_ffmpeg(
                self.temp_filename1, self.filename1, self.true_fps
            )

            if success0 and success1:
                print("Re-encoding complete!")
                try:
                    os.remove(self.temp_filename0)
                    os.remove(self.temp_filename1)
                except:
                    pass
            else:
                print("FFmpeg re-encoding failed. Falling back to rename...")
                self.fallback_rename_temp_files()
        else:
            print("\nFFmpeg not found. Install from ffmpeg.org for best accuracy.")
            print("Falling back to rename...")
            self.fallback_rename_temp_files()

    def fallback_rename_temp_files(self):
        """Fallback: rename temp files"""
        try:
            shutil.move(self.temp_filename0, self.filename0)
            shutil.move(self.temp_filename1, self.filename1)
            print("Videos saved with calibrated FPS.")
        except Exception as e:
            print(f"Error renaming files: {e}")

    def save_frame_data_to_csv(self):
        """Save frame data to CSV files with visibility and stability metrics"""
        header = [
            "Frame Number",
            "Time",
            "Was Duplicated",
            "Valid_Landmark_Percentage",
            "Frame_Valid",
            "neck_angle",
            "neck_reba_score",
            "neck_angle_corrected",
            "neck_reba_score_corrected",
            "neck_angle_natural",
            "neck_reba_score_natural",
            "neck_twisted",
            "neck_side_bent",
            "final_neck_score",
            "final_neck_score_natural",
            "trunk_angle",
            "trunk_reba_score",
            "trunk_angle_corrected",
            "trunk_reba_score_corrected",
            "trunk_angle_natural",
            "trunk_reba_score_natural",
            "trunk_twisted",
            "trunk_side_bent",
            "final_trunk_score",
            "final_trunk_score_natural",
            "leg_raised",
            "left_knee_angle",
            "right_knee_angle",
            "leg_reba_score",
            "leg_raised_corrected",
            "left_knee_angle_corrected",
            "right_knee_angle_corrected",
            "leg_reba_score_corrected",
            "left_knee_angle_natural",
            "right_knee_angle_natural",
            "leg_reba_score_natural",
            "final_leg_score",
            "final_leg_score_natural",
            "table_a_score",
            "table_a_score_natural",
            "force_load_score",
            "score_a",
            "score_a_natural",
            "left_upper_arm_angle",
            "right_upper_arm_angle",
            "upper_arm_reba_score",
            "left_upper_arm_angle_corrected",
            "right_upper_arm_angle_corrected",
            "upper_arm_reba_score_corrected",
            "left_upper_arm_angle_natural",
            "right_upper_arm_angle_natural",
            "upper_arm_reba_score_natural",
            "shoulder_raised",
            "upper_arm_abducted",
            "arm_supported_score",
            "final_upper_arm_score",
            "final_upper_arm_score_natural",
            "left_lower_arm_angle",
            "right_lower_arm_angle",
            "lower_arm_reba_score",
            "left_lower_arm_angle_corrected",
            "right_lower_arm_angle_corrected",
            "lower_arm_reba_score_corrected",
            "left_lower_arm_angle_natural",
            "right_lower_arm_angle_natural",
            "lower_arm_reba_score_natural",
            "left_wrist_angle",
            "right_wrist_angle",
            "wrist_reba_score",
            "left_wrist_angle_corrected",
            "right_wrist_angle_corrected",
            "wrist_reba_score_corrected",
            "left_wrist_angle_natural",
            "right_wrist_angle_natural",
            "wrist_reba_score_natural",
            "left_wrist_bent",
            "right_wrist_bent",
            "wrist_bent_midline_score",
            "left_wrist_twisted",
            "right_wrist_twisted",
            "final_wrist_adjustment_score",
            "final_wrist_score",
            "final_wrist_score_natural",
            "table_b_score",
            "table_b_score_natural",
            "coupling_score",
            "score_b",
            "score_b_natural",
            "table_c_score",
            "table_c_score_natural",
            "activity_score",
            "reba_score",
            "reba_score_natural",
            "action_level",
            "action_level_natural",
            "table_a_score_singlecam",
            "score_a_singlecam",
            "table_b_score_singlecam",
            "score_b_singlecam",
            "table_c_score_singlecam",
            "reba_score_singlecam",
            "action_level_singlecam",
        ]

        for name in self.landmark_names:
            header.append(f"{name}_visibility")

        for name in self.landmark_names:
            header.append(f"{name}_stability")

        header.append("latency_ms")

        # ---------------------------- excluding natural-related & stability  calculations -----------------------------
        # excluding natural-related calculations (comment out this block to include them in csv output
        blank_indices = [i for i, col in enumerate(header) if "natural" in col or "stability" in col]

        def blank_columns(rows):
            for row in rows:
                for idx in blank_indices:
                    if idx < len(row):
                        row[idx] = ""

        blank_columns(self.frame_data0)
        blank_columns(self.frame_data1)

        # ------------------------------------------------------ end ----------------------------------------------------

        with open(self.csv_filename0, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(self.frame_data0)

        with open(self.csv_filename1, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(self.frame_data1)

    def cleanup(self):
        """Clean up resources"""
        self.recording = False

        time.sleep(0.2)

        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()
        if self.out0:
            self.out0.release()
        if self.out1:
            self.out1.release()

        self.pose0.close()
        self.pose1.close()

        cv2.destroyAllWindows()

        self.reencode_videos()

        print("\nSaving frame data to CSV files...")
        self.save_frame_data_to_csv()

        print("\nRecording saved successfully!")
        print(f"  - {self.filename0}")
        print(f"  - {self.filename1}")
        print(f"  - {self.csv_filename0}")
        print(f"  - {self.csv_filename1}")

        print("\n" + "=" * 50)
        print("FRAME STATISTICS")
        print("=" * 50)
        print(f"Total frames captured: {self.total_frames_written}")
        print(f"Camera 0 frames dropped (queue full): {self.dropped_frames0}")
        print(f"Camera 1 frames dropped (queue full): {self.dropped_frames1}")
        print(f"Total frames processed in CSV: {len(self.frame_data0)}")

        if self.total_frames_written != len(self.frame_data0):
            frames_lost = self.total_frames_written - len(self.frame_data0)
            print(f"WARNING: {frames_lost} frames lost during processing")
            if self.dropped_frames0 > 0 or self.dropped_frames1 > 0:
                print(
                    f"  → {max(self.dropped_frames0, self.dropped_frames1)} frames dropped due to queue being full"
                )
                print(
                    f"  → MediaPipe processing was too slow to keep up with capture rate"
                )

        print(f"Camera 0 duplicated frames: {self.duplicated_frames0}")
        print(f"Camera 1 duplicated frames: {self.duplicated_frames1}")

        if self.total_frames_written > 0:
            dup_percent0 = (self.duplicated_frames0 / self.total_frames_written) * 100
            dup_percent1 = (self.duplicated_frames1 / self.total_frames_written) * 100
            drop_percent0 = (self.dropped_frames0 / self.total_frames_written) * 100
            drop_percent1 = (self.dropped_frames1 / self.total_frames_written) * 100
            print(f"Camera 0 duplication rate: {dup_percent0:.2f}%")
            print(f"Camera 1 duplication rate: {dup_percent1:.2f}%")
            print(f"Camera 0 drop rate: {drop_percent0:.2f}%")
            print(f"Camera 1 drop rate: {drop_percent1:.2f}%")

        print("\n" + "-" * 50)
        print("FPS INFORMATION")
        print("-" * 50)
        print(f"Synchronized FPS (from calibration): {self.synchronized_fps:.2f}")
        print(f"True FPS (actual recording): {self.true_fps:.2f}")

        fps_diff_percent = (
            (self.true_fps - self.synchronized_fps) / self.synchronized_fps
        ) * 100
        print(f"FPS difference: {fps_diff_percent:+.2f}%")

        if abs(fps_diff_percent) > 1.0:
            print(f"WARNING: FPS difference > 1% - System may have been under load")

        print("=" * 50)
        print("\nCameras released. Program ended.")


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("DUAL CAMERA CAPTURE WITH MEDIAPIPE POSE DETECTION")
        print("=" * 50)
        print("\nFeatures:")
        print("- Synchronized dual camera recording")
        print("- 90° clockwise rotation applied to both cameras")
        print("- 90° clockwise rotation applied to both cameras")
        print("- MediaPipe pose detection with visibility filtering")
        print("- EMA smoothing (alpha=0.3) on landmark coordinates")
        print("- Frame validation: >=80% landmarks must meet visibility threshold")
        print("- Stability tracking with 10-frame rolling window")
        print("- Landmarks visible in live display only (not in saved videos)")
        print("- Comprehensive CSV logging with visibility and stability metrics")
        print("\n" + "=" * 50)

        dual_cam = DualCameraCapture(cam0_idx=1, cam1_idx=0, fps=30)
        dual_cam.run()

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure both cameras are connected")
        print("2. Close any other applications using the cameras")
        print("3. Try different camera indices if 0 and 1 don't work")
        print("4. Ensure MediaPipe is installed: pip install mediapipe")

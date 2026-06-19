"""
Hand Control Logic - Ported from legacy ROS2 logic without ROS2 dependencies.

This module contains the handlebar-style control logic for computing
robot velocity commands from MediaPipe hand landmarks.
"""

import time
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np

# ============================================================================
# MediaPipe Landmark Indices
# ============================================================================

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger joint tuples
INDEX = (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
MIDDLE = (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
RING = (RING_MCP, RING_PIP, RING_DIP, RING_TIP)
PINKY = (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)


# ============================================================================
# Utility Functions
# ============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def deadzone(x: float, dz: float) -> float:
    """Apply deadzone to input value."""
    if dz >= 1.0:
        return 0.0
    return 0.0 if abs(x) < dz else (x - np.sign(x) * dz) / (1 - dz)


def quantize(x: float, step: float) -> float:
    """Quantize value to discrete steps."""
    if step <= 0:
        return x
    return round(x / step) * step


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


# ============================================================================
# Vector/Geometry Helpers
# ============================================================================

def _v(lm, i: int) -> np.ndarray:
    """Get 3D vector from landmark."""
    p = lm[i]
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def _norm(v: np.ndarray) -> np.ndarray:
    """Normalize vector."""
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed shortest distance a-b in degrees."""
    return ((a - b + 180.0) % 360.0) - 180.0


# ============================================================================
# Hand Geometry Functions
# ============================================================================

def _hand_size_ref(lm) -> float:
    """Get reference hand size for normalization."""
    w = _v(lm, WRIST)
    idx = _v(lm, INDEX_MCP)
    mid = _v(lm, MIDDLE_MCP)
    pink = _v(lm, PINKY_MCP)
    return max(
        (np.linalg.norm(idx - w) + np.linalg.norm(mid - w) + np.linalg.norm(pink - w)) / 3.0,
        1e-3,
    )


def _palm_center(lm) -> np.ndarray:
    """Get center of palm."""
    pts = [
        _v(lm, WRIST),
        _v(lm, INDEX_MCP),
        _v(lm, MIDDLE_MCP),
        _v(lm, RING_MCP),
        _v(lm, PINKY_MCP),
    ]
    return np.mean(pts, axis=0)


def _tip_palm_dist(lm, tip_idx: int) -> float:
    """Get distance from fingertip to palm center."""
    c = _palm_center(lm)
    return float(np.linalg.norm(_v(lm, tip_idx) - c))


def _finger_features(lm, joint_ids: Tuple[int, int, int, int]) -> Dict[str, float]:
    """Extract finger features for gesture recognition."""
    ref = _hand_size_ref(lm)
    MCP, PIP, DIP, TIP = joint_ids
    return {"tip_palm_n": _tip_palm_dist(lm, TIP) / ref}


def _all_finger_feats(lm) -> Dict[str, Dict[str, float]]:
    """Get features for all fingers."""
    return {
        "index": _finger_features(lm, INDEX),
        "middle": _finger_features(lm, MIDDLE),
        "ring": _finger_features(lm, RING),
        "pinky": _finger_features(lm, PINKY),
    }


def _map01(x: float, a0: float, a1: float) -> float:
    """Map value to 0-1 range."""
    return float(np.clip((x - a0) / (a1 - a0), 0.0, 1.0))


def finger_openness_01(f: Dict[str, float]) -> float:
    """Calculate finger openness (0=closed, 1=open)."""
    return _map01(f["tip_palm_n"], 0.50, 0.90)


def per_finger_openness(lm) -> Dict[str, float]:
    """Openness (0=closed, 1=open) per finger: index, middle, ring, pinky.

    Public accessor so discrete-gesture detection (gestures.py) can reuse the
    exact same openness math instead of re-deriving it and drifting from the
    driving logic.
    """
    feats = _all_finger_feats(lm)
    return {name: finger_openness_01(f) for name, f in feats.items()}


def thumb_openness_01(lm) -> float:
    """Thumb extension (0=tucked into fist, 1=sticking out).

    The thumb isn't part of _all_finger_feats (it doesn't matter for the
    handlebar metaphor), so it gets its own mapping. The 0.45..0.75 band is
    tuned for the thumb being a shorter digit than the fingers.
    """
    ref = _hand_size_ref(lm)
    return _map01(_tip_palm_dist(lm, THUMB_TIP) / ref, 0.45, 0.75)


def hand_openness_01(lm) -> float:
    """Calculate overall hand openness."""
    vals = list(per_finger_openness(lm).values())
    return float(np.mean(vals)) if vals else 0.0


# ============================================================================
# Handlebar Metrics
# ============================================================================

def compute_handlebar_metrics(lm_left, lm_right) -> Dict[str, Any]:
    """Calculate handlebar metrics from both hands."""
    left_palm = _palm_center(lm_left)
    right_palm = _palm_center(lm_right)
    vec_lr = right_palm - left_palm

    span_xy = float(np.linalg.norm(vec_lr[:2]))
    span_3d = float(np.linalg.norm(vec_lr))
    mean_z = float((left_palm[2] + right_palm[2]) / 2.0)
    depth_diff = float(right_palm[2] - left_palm[2])  # >0: right farther than left
    size_avg = float((_hand_size_ref(lm_left) + _hand_size_ref(lm_right)) / 2.0)
    angle_deg = math.degrees(math.atan2(vec_lr[1], vec_lr[0]))

    return {
        'left_palm': left_palm,
        'right_palm': right_palm,
        'span_xy': span_xy,
        'span_3d': span_3d,
        'mean_z': mean_z,
        'depth_diff': depth_diff,
        'size_avg': size_avg,
        'angle_deg': angle_deg,
    }


# ============================================================================
# Drive State Machine
# ============================================================================

class DriveState:
    """
    State machine for handlebar-style drive control.
    
    Manages:
    - Enable/disable toggle via both-hands-open gesture
    - Calibration of neutral position
    - Smoothing and quantization of control outputs
    - Visibility/lost-hands safety handling
    """

    def __init__(self):
        # Enable state
        self.enabled = False
        self._arming = False
        self._t0: Optional[float] = None
        self._hold = 1.5  # seconds to hold both hands open for toggle

        # Control outputs (normalized -1 to +1)
        self.speed = 0.0
        self.direction = 0.0
        self._f_speed = 0.0
        self._f_dir = 0.0

        # Quantized command outputs
        self.speed_cmd = 0
        self.direction_cmd = 0
        self.quant_step = 0.1  # Integer steps -10..10
        self.command_hysteresis = 0.6  # Sticky to prevent flicker
        self.zero_lock = 0.03  # Deadzone near zero

        # Calibration state
        self.baseline: Optional[Dict[str, Any]] = None
        self.calibrated = False
        self.calibrating = False
        self.calib_progress = 0.0
        self.calib_hold = 0.8  # seconds of stability to lock origin
        self.calib_stable_norm_thr = 0.08
        self.prev_metrics: Optional[Dict[str, Any]] = None
        self.calib_requested = False

        # Visibility/safety state
        self._last_full_seen = time.time()
        self._lost = False
        self.fail_timeout = 0.4  # seconds before applying safety decay
        self.decay_rate = 6.0  # how fast to decay when lost
        self.metrics_lp: Optional[Dict[str, Any]] = None
        self.metrics_smoothing = 6.0

        # Ranges for bike-like control
        self.depth_range_forward = 0.05  # Push hands toward camera for forward
        self.depth_range_reverse = 0.01  # Pull hands back for reverse (more sensitive)
        self.size_range = 0.30           # How much to spread hands for speed boost
        self.turn_depth_range = 0.06     # Steering requires larger hand difference

    def update_arming(self, both_open: bool) -> Optional[float]:
        """
        Update arming state based on whether both hands are open.
        
        Returns progress (0-1) if arming, None otherwise.
        """
        if both_open:
            if not self._arming:
                self._arming = True
                self._t0 = time.time()
            elapsed = time.time() - self._t0
            if elapsed >= self._hold:
                self.enabled = not self.enabled
                self._arming = False
                self._t0 = None
                return None
            return clamp(elapsed / self._hold, 0.0, 1.0)
        else:
            self._arming = False
            self._t0 = None
            return None

    def reset_calibration(self) -> None:
        """Reset calibration state."""
        self.baseline = None
        self.calibrated = False
        self.calibrating = False
        self.calib_progress = 0.0
        self.prev_metrics = None
        self.metrics_lp = None

    def _stable_enough(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics are stable enough for calibration."""
        if metrics is None:
            return False
        if self.prev_metrics is None:
            self.prev_metrics = metrics
            return False
            
        prev = self.prev_metrics
        keys = ['span_xy', 'mean_z', 'depth_diff', 'angle_deg', 'size_avg']
        norm = {
            'span_xy': max(prev['span_xy'], metrics['span_xy'], 1e-3),
            'mean_z': 0.05,
            'depth_diff': 0.05,
            'angle_deg': 60.0,
            'size_avg': max(prev['size_avg'], metrics['size_avg'], 1e-3),
        }
        diff = 0.0
        count = 0
        for k in keys:
            dv = abs(metrics[k] - prev[k])
            diff += dv / norm[k]
            count += 1
        diff = diff / max(count, 1)
        self.prev_metrics = metrics
        return diff < self.calib_stable_norm_thr

    def filter_metrics(self, metrics: Optional[Dict[str, Any]], dt: float) -> Optional[Dict[str, Any]]:
        """Low-pass filter noisy handlebar metrics."""
        if metrics is None:
            self.metrics_lp = None
            return None

        if self.metrics_lp is None:
            self.metrics_lp = metrics.copy()
            return metrics

        alpha = clamp(dt * self.metrics_smoothing, 0.0, 1.0)
        out = metrics.copy()
        for k in ('span_xy', 'mean_z', 'depth_diff', 'size_avg'):
            prev = self.metrics_lp.get(k, metrics[k])
            out[k] = float(lerp(prev, metrics[k], alpha))

        # angle needs wrap-aware interpolation
        prev_ang = self.metrics_lp.get('angle_deg', metrics['angle_deg'])
        ang_delta = _angle_diff_deg(metrics['angle_deg'], prev_ang)
        out['angle_deg'] = float(prev_ang + ang_delta * alpha)

        self.metrics_lp = out
        return out

    def calibration_tick(
        self, 
        metrics: Optional[Dict[str, Any]], 
        dt: float, 
        requested: bool = False
    ) -> Dict[str, Any]:
        """Process calibration state machine."""
        if metrics is None:
            self.calibrating = False
            self.calib_progress = 0.0
            self.prev_metrics = None
            return {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}

        should_start = requested or (self.baseline is None)
        if should_start and not self.calibrating:
            self.calibrating = True
            self.calib_progress = 0.0

        if not self.calibrating:
            return {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}

        stable = self._stable_enough(metrics)
        if stable:
            self.calib_progress = clamp(self.calib_progress + dt / self.calib_hold, 0.0, 1.0)
        else:
            self.calib_progress = 0.0

        if self.calib_progress >= 1.0:
            self.baseline = metrics.copy()
            self.calibrated = True
            self.calibrating = False
            self.calib_progress = 0.0
            return {'calibrating': False, 'progress': 1.0, 'calibrated_now': True}

        return {'calibrating': True, 'progress': self.calib_progress, 'calibrated_now': False}

    def smooth_and_quantize(self, spd: float, direc: float, dt: float) -> None:
        """Smooth and quantize speed and direction values."""
        t = clamp(dt * 6.0, 0.0, 1.0)
        self._f_speed = lerp(self._f_speed, spd, t)
        self._f_dir = lerp(self._f_dir, direc, t)

        if abs(self._f_speed) < self.zero_lock:
            self._f_speed = 0.0
        if abs(self._f_dir) < self.zero_lock:
            self._f_dir = 0.0

        target_speed = quantize(clamp(self._f_speed, -1.0, 1.0), self.quant_step)
        target_dir = quantize(clamp(self._f_dir, -1.0, 1.0), self.quant_step)

        stick = self.quant_step * self.command_hysteresis
        if abs(target_speed - self.speed) < stick:
            target_speed = self.speed
        if abs(target_dir - self.direction) < stick:
            target_dir = self.direction

        self.speed = target_speed
        self.direction = target_dir

        self.speed_cmd = int(round(self.speed / self.quant_step))
        self.direction_cmd = int(round(self.direction / self.quant_step))

    def handle_visibility(self, hands_ok: bool, dt: float) -> bool:
        """
        Handle hand visibility loss with safety decay.
        
        Returns True if currently in lost/decay state.
        """
        now = time.time()
        if hands_ok:
            self._last_full_seen = now
            self._lost = False
            return False

        if now - self._last_full_seen >= self.fail_timeout:
            decay = math.exp(-self.decay_rate * dt)
            self._f_speed *= decay
            self._f_dir *= decay
            self.speed = quantize(self._f_speed, self.quant_step)
            self.direction = quantize(self._f_dir, self.quant_step)
            self.speed_cmd = int(round(self.speed / self.quant_step))
            self.direction_cmd = int(round(self.direction / self.quant_step))
            self._lost = True
            self.metrics_lp = None
            return True
        return False

    def force_stop(self) -> None:
        """Force all outputs to zero immediately."""
        self._f_speed = 0.0
        self._f_dir = 0.0
        self.speed = 0.0
        self.direction = 0.0
        self.speed_cmd = 0
        self.direction_cmd = 0

    @property
    def is_lost(self) -> bool:
        """Check if hands are currently lost."""
        return self._lost

    def get_velocity_commands(
        self, 
        max_linear: float, 
        max_angular: float
    ) -> Tuple[float, float]:
        """
        Get scaled velocity commands.
        
        Returns (linear_vel, angular_vel) scaled to max values.
        """
        if not self.enabled:
            return 0.0, 0.0
            
        linear = self.speed_cmd * self.quant_step * max_linear
        angular = -self.direction_cmd * self.quant_step * max_angular
        return linear, angular


# ============================================================================
# Bike-like Control Computation
# ============================================================================

def compute_bike_controls(
    lm_left,
    lm_right,
    state: DriveState,
    metrics: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
    """
    Compute bike-like controls from two hands.
    
    Args:
        lm_left: Left hand landmarks
        lm_right: Right hand landmarks
        state: DriveState instance
        metrics: Pre-computed handlebar metrics (optional)
        
    Returns:
        Tuple of (speed_raw, steering_raw, metrics, info)
    """
    metrics = metrics if metrics is not None else compute_handlebar_metrics(lm_left, lm_right)
    
    if state.baseline is None:
        return 0.0, 0.0, metrics, {'calibrating': True}

    base = state.baseline
    info = {'calibrating': False}

    # Speed: push both hands forward / pull back
    # Positive depth_delta = hands closer to camera = forward
    # Negative depth_delta = hands farther from camera = reverse
    depth_delta = base['mean_z'] - metrics['mean_z']
    
    # Asymmetric sensitivity: reverse is more sensitive
    if depth_delta >= 0:
        # Forward
        depth_push = clamp(depth_delta / state.depth_range_forward, 0.0, 1.0)
    else:
        # Reverse (more sensitive)
        depth_push = clamp(depth_delta / state.depth_range_reverse, -1.0, 0.0)
    
    size_push = clamp(
        (metrics['size_avg'] - base['size_avg']) / (base['size_avg'] * state.size_range + 1e-6),
        -0.3, 0.3  # Secondary effect, limited range
    )

    # Speed is primarily depth-based
    speed_combined = depth_push + size_push
    # Apply deadzone to reject noise
    speed_raw = clamp(deadzone(speed_combined, 0.08), -1.0, 1.0)

    # Steering: left hand forward / right hand back
    steer_depth = clamp(metrics['depth_diff'] / state.turn_depth_range, -1.0, 1.0)
    ang_delta = _angle_diff_deg(metrics['angle_deg'], base['angle_deg'])
    steer_angle = clamp(ang_delta / 60.0, -1.0, 1.0)  # Less sensitive to tilt

    # Steering is primarily depth-based (one hand forward, one back)
    steering_combined = 0.85 * steer_depth + 0.15 * steer_angle
    # LARGE deadzone: need deliberate movement to steer (default is straight)
    steering_raw = clamp(deadzone(steering_combined, 0.25), -1.0, 1.0)

    info.update({
        'depth_push': depth_push,
        'size_push': size_push,
        'steer_depth': steer_depth,
        'steer_angle': steer_angle,
        'baseline_angle': base['angle_deg'],
    })
    
    return speed_raw, steering_raw, metrics, info


# ============================================================================
# Hand Detection Helper
# ============================================================================

def extract_hands(results) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Extract left and right hands from MediaPipe results.
    
    Args:
        results: MediaPipe hands processing results
        
    Returns:
        Tuple of (left_hand, right_hand) where each is (landmarks, label) or None
    """
    left = None
    right = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hd.classification[0].label
            if label == "Left" and left is None:
                left = (lm, label)
            elif label == "Right" and right is None:
                right = (lm, label)
                
    return left, right


def check_both_hands_open(left, right, threshold: float = 0.85) -> bool:
    """Check if both hands are detected and open."""
    if left is None or right is None:
        return False
        
    left_open = hand_openness_01(left[0].landmark) >= threshold
    right_open = hand_openness_01(right[0].landmark) >= threshold
    
    return left_open and right_open


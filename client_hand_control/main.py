#!/usr/bin/env python3
"""
Hand Gesture Control Client - Main Entry Point

This client runs on a user laptop, processes camera/RTSP frames locally
with MediaPipe, and fires validated JSON control messages straight at the
rover as UDP datagrams. The rover (purePursuit.py) listens and acts.

NO ROS2 DEPENDENCIES. NO BROKER.

Usage:
    python -m client_hand_control.main --camera 0 --preview
    ROBOT_IP=10.42.0.243 python -m client_hand_control.main --camera 0 --preview
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .frame_gate import FrameGate, MediaPipeGate
from .hand_control import (
    DriveState,
    compute_bike_controls,
    compute_handlebar_metrics,
    extract_hands,
    check_both_hands_open,
)
from .gestures import GestureState
from .udp_publisher import UdpPublisher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class HandControlClient:
    """
    Main client that integrates all components:
    - Camera/RTSP capture
    - Frame quality gate
    - MediaPipe hand detection
    - Handlebar control logic
    - Message validation
    - WebSocket communication
    """
    
    def __init__(
        self,
        host: str,
        port: int = 5001,
        camera_index: int = 0,
        rtsp_url: Optional[str] = None,
        max_linear: float = 0.22,
        max_angular: float = 2.84,
        rate: float = 25.0,  # 15 Hz is plenty for robot control
        show_preview: bool = False,
        invalid_timeout_ms: int = 300,
    ):
        """
        Initialize the hand control client.

        Args:
            host: Rover IP/host to send UDP control datagrams to
            port: Rover UDP port
            camera_index: Camera device index (used if rtsp_url is None)
            rtsp_url: RTSP stream URL (overrides camera_index if set)
            max_linear: Maximum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            rate: Control loop rate (Hz)
            show_preview: Whether to show OpenCV preview window
            invalid_timeout_ms: Time before force-stop on invalid frames
        """
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.rtsp_url = rtsp_url
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.rate = rate
        self.show_preview = show_preview
        self.invalid_timeout_ms = invalid_timeout_ms

        # Components
        self.frame_gate = FrameGate(invalid_timeout_ms=invalid_timeout_ms)
        self.mp_gate = MediaPipeGate()
        self.drive_state = DriveState()
        # fist_thresh: four fingers must be folded below this. open_thresh: thumb
        # extension above this counts as "thumb out" (thumbs-up, and the line that
        # separates a fist from a thumbs-up). See is_fist/is_thumb_up in gestures.py.
        self.gesture_state = GestureState(fist_thresh=0.35, open_thresh=0.55)
        self.publisher: Optional[UdpPublisher] = None
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        
        # MediaPipe
        self.hands: Optional[mp_hands.Hands] = None
        
        # State
        self._running = False
        self._prev_time = time.time()
        self._last_send_time = 0.0
        self._send_interval = 1.0 / rate
        self._force_stop_sent = False

        # UI font + HUD scale. Bump hud_scale to make all overlay text bigger.
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.hud_scale = 1.8

        # Keyboard state. Single input path: cv2.waitKey in the preview loop.
        # (pynput was removed — it duplicated every keypress and needs macOS
        # accessibility perms. Click the preview window to give it focus.)
        self._last_key_pressed = None

    def _hud(self, frame, text, pos, scale, color, thick=2):
        """Draw overlay text scaled by self.hud_scale (anti-aliased)."""
        s = scale * self.hud_scale
        t = max(1, int(round(thick * self.hud_scale)))
        cv2.putText(frame, text, pos, self.font, s, color, t, cv2.LINE_AA)

    def _send_raw_command(self, cmd: str):
        """Send a raw command string directly to the rover."""
        if self.publisher:
            self.publisher.publish(cmd)
            logger.debug(f"KB: {cmd}")

    async def start(self) -> None:
        """Start the client."""
        logger.info("Starting Hand Control Client...")
        
        # Initialize camera
        if not self._init_camera():
            raise RuntimeError("Failed to initialize camera")
            
        # Initialize MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Initialize UDP publisher (fires datagrams straight at the rover)
        self.publisher = UdpPublisher(self.host, self.port)
        self.publisher.start()

        self._running = True
        logger.info("Hand Control Client started")

    async def stop(self) -> None:
        """Stop the client and clean up resources."""
        logger.info("Stopping Hand Control Client...")
        self._running = False

        # Send a final STOP (raw string the rover understands), then shut down.
        if self.publisher:
            self.publisher.publish("CMD_S")
            self.publisher.stop()

        # Clean up camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Clean up MediaPipe
        if self.hands:
            self.hands.close()
            self.hands = None

        # Clean up OpenCV windows
        if self.show_preview:
            cv2.destroyAllWindows()

        logger.info("Hand Control Client stopped")
        
    async def run(self) -> None:
        """Main control loop."""
        target_dt = 1.0 / self.rate
        
        while self._running:
            loop_start = time.time()
            
            try:
                await self._process_frame()
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                
            # Handle OpenCV window events
            if self.show_preview:
                key = cv2.waitKey(1) & 0xFF
                # Show what was pressed in the overlay (255 = no key this frame).
                key_names = {
                    ord(' '): "SPACE", ord('m'): "M", ord('e'): "E", ord('c'): "C",
                    ord('w'): "W", ord('a'): "A", ord('s'): "S", ord('d'): "D",
                    82: "UP", 84: "DOWN", 81: "LEFT", 83: "RIGHT",
                }
                if key != 255:
                    self._last_key_pressed = key_names.get(key, chr(key) if 32 <= key < 127 else str(key))
                if key in (27, ord('q')):
                    logger.info("Quit requested")
                    self._running = False
                elif key == ord(' '):
                    # SPACE = emergency stop toggle. Keyboard, not a gesture: a
                    # vision e-stop fails exactly when vision fails.
                    self.gesture_state.estop = not self.gesture_state.estop
                    self.gesture_state.last_event = "E-STOP" if self.gesture_state.estop else "E-STOP CLEARED"
                    if self.gesture_state.estop:
                        # Actually stop the rover, not just our local flag. This
                        # rover has no ESTOP command; CMD_S stops it in ON_ROAD.
                        self.gesture_state.run = False
                        self._send_raw_command("CMD_S")
                    logger.warning(f"E-STOP {'ENGAGED' if self.gesture_state.estop else 'RELEASED'} (keyboard)")
                elif key in (ord('m'), ord('M')):
                    # M = mode toggle (ON_ROAD <-> OFF_ROAD)
                    self.gesture_state.mode = "onroad" if self.gesture_state.mode == "offroad" else "offroad"
                    self.gesture_state.last_event = f"MODE {self.gesture_state.mode.upper()}"
                    self._send_raw_command("MODE_TOGGLE")
                    logger.info(f"Mode toggle: {self.gesture_state.mode}")
                elif key in (ord('e'), ord('E')):
                    self.drive_state.enabled = not self.drive_state.enabled
                    if self.drive_state.enabled:
                        self.gesture_state.estop = False  # enabling drive clears the latch
                    logger.info(f"Drive {'ENABLED' if self.drive_state.enabled else 'DISABLED'} (keyboard)")
                elif key in (ord('c'), ord('C')):
                    self.drive_state.reset_calibration()
                    self.drive_state.calib_requested = True
                    logger.info("Calibration reset requested")
                elif key == ord('w'):
                    self._send_raw_command("CMD_W")
                elif key == ord('a'):
                    self._send_raw_command("CMD_A")
                elif key == ord('s'):
                    self._send_raw_command("CMD_S")
                elif key == ord('d'):
                    self._send_raw_command("CMD_D")
                elif key in (82, 81):  # Up / Left arrow -> lane left
                    self._send_raw_command("CMD_A")
                elif key in (84, 83):  # Down / Right arrow -> lane right
                    self._send_raw_command("CMD_D")
                    
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                await asyncio.sleep(target_dt - elapsed)
                
    async def _process_frame(self) -> None:
        """Process a single frame through the pipeline."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        
        # Read frame from camera
        ok, frame = self.cap.read()
        
        # ====== FRAME QUALITY GATE ======
        frame_result = self.frame_gate.validate(ok, frame)
        
        if not frame_result.valid:
            logger.debug(f"Frame invalid: {frame_result.reason}")
            
            # Check if we should force stop
            if self.frame_gate.should_force_stop():
                await self._send_force_stop()
            return
            
        frame = frame_result.frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ====== MEDIAPIPE PROCESSING ======
        mp_ok, results = self.mp_gate.process(self.hands, rgb)
        
        if not mp_ok:
            logger.debug("MediaPipe processing failed")
            if self.mp_gate.is_stream_problematic():
                logger.warning("Stream appears problematic, forcing stop")
                await self._send_force_stop()
            return
            
        # Extract hand landmarks
        left, right = extract_hands(results)
        hands_ok = (left is not None) and (right is not None)

        # ====== ARMING GESTURE (both hands open) ======
        both_open = check_both_hands_open(left, right)
        prev_enabled = self.drive_state.enabled
        arming_progress = self.drive_state.update_arming(both_open)
        # A rising edge on `enabled` means we just (re-)armed; that also clears a
        # latched e-stop, so there's no separate "unstop" gesture.
        rearmed = self.drive_state.enabled and not prev_enabled

        # ====== DISCRETE GESTURE COMMANDS (mode / lane / e-stop) ======
        gesture_result = self.gesture_state.update(left, right, dt, rearmed=rearmed)
        # Forward any fired gesture as a raw string command the rover understands
        # (MODE_TOGGLE / CMD_W / CMD_S / CMD_A / CMD_D). This is the ONLY thing the
        # rover acts on — it parses plain strings, not JSON.
        for cmd in gesture_result.get("commands", []):
            self._send_raw_command(cmd)
        
        # ====== HAND QUALITY GATE + CONTROL COMPUTATION ======
        lost_active = self.drive_state.handle_visibility(hands_ok, dt)
        
        # Queue recalibration if hands lost for too long
        if lost_active and left is None and right is None and not self.drive_state.calib_requested:
            self.drive_state.calib_requested = True
            
        metrics = None
        calib_info = {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}
        
        if hands_ok:
            # Compute handlebar metrics
            metrics_raw = compute_handlebar_metrics(left[0].landmark, right[0].landmark)
            metrics = self.drive_state.filter_metrics(metrics_raw, dt)
            
            # Calibration tick
            calib_info = self.drive_state.calibration_tick(
                metrics, dt, 
                requested=self.drive_state.calib_requested
            )
            if calib_info.get('calibrated_now'):
                self.drive_state.calib_requested = False
                logger.info("Calibration complete")
                
            # Compute controls if calibrated and not calibrating
            if self.drive_state.baseline is not None and not self.drive_state.calibrating:
                speed_raw, dir_raw, _, _ = compute_bike_controls(
                    left[0].landmark, right[0].landmark,
                    self.drive_state, metrics=metrics
                )
                self.drive_state.smooth_and_quantize(speed_raw, dir_raw, dt)
            else:
                self.drive_state.smooth_and_quantize(0.0, 0.0, dt)
        else:
            # No hands detected
            self.drive_state.filter_metrics(None, dt)
            self.drive_state.calibration_tick(None, dt, requested=False)
            if not lost_active:
                self.drive_state.smooth_and_quantize(0.0, 0.0, dt)
                
        # NOTE: there is no periodic message stream. The rover acts on discrete
        # string commands only (sent above on gesture edges, and from the keyboard).
        # In ON_ROAD the rover latches — CMD_W keeps it lane-following until CMD_S,
        # so re-sending state every frame would just re-trigger lane changes.
            
        # ====== PREVIEW DISPLAY ======
        if self.show_preview:
            self._draw_preview(frame, h, w, left, right, arming_progress, 
                             both_open, hands_ok, calib_info, lost_active)
            cv2.imshow("Hand Control Client", frame)
            
    async def _send_force_stop(self) -> None:
        """Stop the rover due to frame quality issues (dead/corrupted camera).

        The rover only understands raw string commands and has no ESTOP, so we
        send CMD_S — the stop it acts on in ON_ROAD.
        """
        if self._force_stop_sent:
            return

        logger.warning("Sending force stop due to frame quality issues")

        # Force drive state to stop
        self.drive_state.force_stop()
        self.drive_state.enabled = False
        self.gesture_state.run = False

        if self.publisher and self.publisher.publish("CMD_S"):
            self._force_stop_sent = True
            logger.info("Force stop (CMD_S) sent")
                
    def _init_camera(self) -> bool:
        """Initialize video capture."""
        if self.rtsp_url:
            logger.info(f"Opening RTSP stream: {self.rtsp_url}")
            # For RTSP, prefer TCP transport for reliability
            # Add ?rtsp_transport=tcp if not already present
            url = self.rtsp_url
            if '?' not in url:
                url += '?rtsp_transport=tcp'
            elif 'rtsp_transport' not in url:
                url += '&rtsp_transport=tcp'
            self.cap = cv2.VideoCapture(url)
        else:
            logger.info(f"Opening camera index: {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            logger.error("Failed to open camera source")
            return False
            
        # Get and log camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {width}x{height} @ {fps:.1f} fps")
        
        return True

    def _draw_preview(
        self, 
        frame: np.ndarray, 
        h: int, 
        w: int,
        left, 
        right,
        arming_progress: Optional[float],
        both_open: bool,
        hands_ok: bool,
        calib_info: dict,
        lost_active: bool,
    ) -> None:
        """Draw preview overlay."""
        # Draw hand landmarks
        if left:
            mp_draw.draw_landmarks(frame, left[0], mp_hands.HAND_CONNECTIONS)
        if right:
            mp_draw.draw_landmarks(frame, right[0], mp_hands.HAND_CONNECTIONS)
            
        # Status text
        status = f"Drive {'ENABLED' if self.drive_state.enabled else 'DISABLED'}"
        
        # Color based on state
        if arming_progress is not None and both_open:
            future_enabled = not self.drive_state.enabled
            color = (0, 255, 0) if future_enabled else (0, 0, 255)
        else:
            color = (0, 255, 0) if self.drive_state.enabled else (0, 0, 255)
            
        self._hud(frame, status, (20, 55), 0.5, color)
        
        # UDP target status. Note: UDP is connectionless — "Ready" just means the
        # socket is open, NOT that the rover is actually receiving. There's no ack.
        conn_status = "Ready" if (self.publisher and self.publisher.connected) else "Down"
        conn_color = (0, 255, 0) if conn_status == "Ready" else (0, 0, 255)
        self._hud(frame, f"Robot {self.host}:{self.port} [{conn_status}]", (20, 100), 0.32, conn_color)

        # Discrete-gesture state: mode / run / lane / e-stop
        gs = self.gesture_state
        mode_color = (0, 200, 255) if gs.mode == "onroad" else (180, 180, 180)
        self._hud(frame, f"Mode: {gs.mode.upper()}", (20, 150), 0.5, mode_color)

        run_color = (0, 255, 0) if gs.run else (0, 0, 255)
        self._hud(frame, f"RUN: {'ON' if gs.run else 'OFF'}", (20, 200), 0.5, run_color)

        # Gesture detection debug: show what poses are detected
        if hands_ok and left and right:
            from .gestures import is_fist, is_thumb_up, is_index_up, thumb_direction
            from .hand_control import (
                hand_openness_01, thumb_openness_01, THUMB_MCP, THUMB_TIP,
            )
            l_lm = left[0].landmark
            r_lm = right[0].landmark

            l_open = hand_openness_01(l_lm)
            r_open = hand_openness_01(r_lm)
            l_thumb_open = thumb_openness_01(l_lm)
            r_thumb_open = thumb_openness_01(r_lm)

            l_fist = is_fist(l_lm, gs.fist_thresh, gs.open_thresh)
            r_fist = is_fist(r_lm, gs.fist_thresh, gs.open_thresh)
            l_thumb = is_thumb_up(l_lm, gs.fist_thresh, gs.open_thresh)
            r_thumb = is_thumb_up(r_lm, gs.fist_thresh, gs.open_thresh)
            l_idx = is_index_up(l_lm, gs.fist_thresh)
            r_idx = is_index_up(r_lm, gs.fist_thresh)

            def _tag(fist, idx, thumb):
                return 'F' if fist else 'I' if idx else 'T' if thumb else '.'

            gesture_text = f"L:{_tag(l_fist, l_idx, l_thumb)} R:{_tag(r_fist, r_idx, r_thumb)}  (II=mode)"
            self._hud(frame, gesture_text, (w - 280, 55), 0.45, (200, 200, 200))

            # Show actual values
            openness_text = f"L: {l_open:.2f} ({l_thumb_open:.2f}) R: {r_open:.2f} ({r_thumb_open:.2f})"
            self._hud(frame, openness_text, (w - 520, 100), 0.3, (150, 150, 150))

            # Show thresholds
            thresh_text = f"Fist<{gs.fist_thresh:.2f} Thumb>{gs.open_thresh:.2f}"
            self._hud(frame, thresh_text, (w - 520, 135), 0.3, (100, 100, 100))

            # Lane (thumb-sideways) debug: per-hand thumb dx/dy + detected dir.
            l_dir = thumb_direction(l_lm, gs.lane_dx, gs.fist_thresh, gs.lane_thumb_open)
            r_dir = thumb_direction(r_lm, gs.lane_dx, gs.fist_thresh, gs.lane_thumb_open)
            l_dxy = (l_lm[THUMB_TIP].x - l_lm[THUMB_MCP].x, l_lm[THUMB_TIP].y - l_lm[THUMB_MCP].y)
            r_dxy = (r_lm[THUMB_TIP].x - r_lm[THUMB_MCP].x, r_lm[THUMB_TIP].y - r_lm[THUMB_MCP].y)
            lane_color = (0, 255, 0) if (l_dir or r_dir) else (120, 120, 120)
            self._hud(frame, f"LANE L[{l_dir or '-'}] R[{r_dir or '-'}]", (w - 520, 175), 0.35, lane_color)
            self._hud(frame, f"Ldx{l_dxy[0]:+.2f} dy{l_dxy[1]:+.2f} to{l_thumb_open:.2f}", (w - 520, 210), 0.3, (150, 150, 150))
            self._hud(frame, f"Rdx{r_dxy[0]:+.2f} dy{r_dxy[1]:+.2f} to{r_thumb_open:.2f}", (w - 520, 245), 0.3, (150, 150, 150))

        # Lane change state
        if gs.lane_change:
            lane_text = f"Lane: {gs.lane_change.upper()} (seq={gs.lane_seq})"
            self._hud(frame, lane_text, (20, 250), 0.42, (255, 200, 0))

        if gs.estop:
            # Loud, centered banner — this is the one you must not miss.
            self._hud(frame, "*** E-STOP ***", (w // 2 - 220, 80), 1.0, (0, 0, 255), 3)

        if gs.last_event:
            self._hud(frame, gs.last_event, (20, 300), 0.42, (0, 255, 0))
        
        # Keyboard display
        if self._last_key_pressed:
            self._hud(frame, f"Key: {self._last_key_pressed}", (20, h - 200), 0.5, (255, 165, 0))

        # Command display
        linear, angular = self.drive_state.get_velocity_commands(self.max_linear, self.max_angular)
        self._hud(frame, f"Linear: {linear:+.3f} m/s", (20, h - 90), 0.45, (255, 0, 0))
        self._hud(frame, f"Angular: {angular:+.3f} rad/s", (20, h - 40), 0.45, (0, 255, 255))
        
        # Status messages
        if calib_info.get('calibrating'):
            progress = int(calib_info.get('progress', 0) * 100)
            self._hud(frame, f"Calibrating... {progress}%", (20, h - 145), 0.42, (0, 200, 0))
        elif self.drive_state.calib_requested:
            self._hud(frame, "Hold both hands still to calibrate", (20, h - 145), 0.42, (0, 165, 255))
        elif self.drive_state.baseline is None:
            self._hud(frame, "Show both hands to calibrate", (20, h - 145), 0.42, (0, 165, 255))
        elif lost_active:
            self._hud(frame, "SAFE STOP (hands lost)", (20, h - 145), 0.42, (0, 0, 255))
        elif not hands_ok:
            self._hud(frame, "Need both hands visible", (20, h - 145), 0.42, (0, 165, 255))
            
        # Frame gate stats
        fg_stats = self.frame_gate.get_stats()
        if fg_stats['invalid_frames'] > 0:
            invalid_pct = fg_stats['invalid_frames'] / max(fg_stats['total_frames'], 1) * 100
            self._hud(frame, f"Frame errors: {invalid_pct:.1f}%", (w - 280, 290), 0.35, (0, 0, 255))


async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point."""
    client = HandControlClient(
        host=args.robot_ip,
        port=args.port,
        camera_index=args.camera,
        rtsp_url=args.rtsp,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        rate=args.rate,
        show_preview=args.preview,
        invalid_timeout_ms=args.invalid_timeout,
    )
    
    # Handle shutdown signals (Unix only - Windows uses KeyboardInterrupt)
    import sys
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()
        
        def signal_handler():
            logger.info("Shutdown signal received")
            asyncio.create_task(client.stop())
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                pass  # Windows doesn't support signal handlers in asyncio
        
    try:
        await client.start()
        await client.run()
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await client.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hand Gesture Control Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=os.environ.get("ROBOT_IP", "127.0.0.1"),
        help="Rover IP/host to send UDP control datagrams to (env: ROBOT_IP)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ROBOT_PORT", "5001")),
        help="Rover UDP port (env: ROBOT_PORT)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--rtsp",
        type=str,
        default=None,
        help="RTSP URL (overrides --camera if set)",
    )
    parser.add_argument(
        "--max-linear",
        type=float,
        default=0.22,
        help="Maximum linear velocity (m/s)",
    )
    parser.add_argument(
        "--max-angular",
        type=float,
        default=2.84,
        help="Maximum angular velocity (rad/s)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=15.0,
        help="Control loop rate (Hz) - 15 Hz is recommended",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window",
    )
    parser.add_argument(
        "--invalid-timeout",
        type=int,
        default=300,
        help="Timeout (ms) before force-stop on invalid frames",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()


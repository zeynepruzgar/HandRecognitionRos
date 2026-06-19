#!/usr/bin/env python3
"""
Hand Gesture Control Client - Main Entry Point

This client runs on a user laptop, processes camera/RTSP frames locally
with MediaPipe, and sends validated JSON control messages to a server
gateway over WebSocket.

NO ROS2 DEPENDENCIES.

Usage:
    python -m client_hand_control.main --server ws://127.0.0.1:8080/control --token SECRET --camera 0
    python -m client_hand_control.main --server ws://127.0.0.1:8080/control --token SECRET --rtsp rtsp://10.8.34.150:8554/handcam
"""

import argparse
import asyncio
import logging
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
from .message import ControlMessage, MessageValidator, create_control_message
from .ws_client import WebSocketClient

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
        server_url: str,
        token: str,
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
            server_url: WebSocket server URL
            token: Authentication token
            camera_index: Camera device index (used if rtsp_url is None)
            rtsp_url: RTSP stream URL (overrides camera_index if set)
            max_linear: Maximum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            rate: Control loop rate (Hz)
            show_preview: Whether to show OpenCV preview window
            invalid_timeout_ms: Time before force-stop on invalid frames
        """
        self.server_url = server_url
        self.token = token
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
        self.gesture_state = GestureState()
        self.validator = MessageValidator(max_linear=max_linear, max_angular=max_angular)
        self.ws_client: Optional[WebSocketClient] = None
        
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

        # UI font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
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
        
        # Initialize WebSocket client
        self.ws_client = WebSocketClient(
            server_url=self.server_url,
            token=self.token,
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
        )
        await self.ws_client.start()
        
        self._running = True
        logger.info("Hand Control Client started")
        
    async def stop(self) -> None:
        """Stop the client and clean up resources."""
        logger.info("Stopping Hand Control Client...")
        self._running = False
        
        # Send final STOP
        if self.ws_client:
            await self.ws_client.stop()
            
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
                if key in (27, ord('q')):
                    logger.info("Quit requested")
                    self._running = False
                elif key == ord(' '):
                    # SPACE = emergency stop toggle. Keyboard, not a gesture: a
                    # vision e-stop fails exactly when vision fails.
                    self.gesture_state.estop = not self.gesture_state.estop
                    self.gesture_state.last_event = "E-STOP" if self.gesture_state.estop else "E-STOP CLEARED"
                    logger.warning(f"E-STOP {'ENGAGED' if self.gesture_state.estop else 'RELEASED'} (keyboard)")
                elif key in (ord('e'), ord('E')):
                    self.drive_state.enabled = not self.drive_state.enabled
                    if self.drive_state.enabled:
                        self.gesture_state.estop = False  # enabling drive clears the latch
                    logger.info(f"Drive {'ENABLED' if self.drive_state.enabled else 'DISABLED'} (keyboard)")
                elif key in (ord('c'), ord('C')):
                    self.drive_state.reset_calibration()
                    self.drive_state.calib_requested = True
                    logger.info("Calibration reset requested")
                    
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
        self.gesture_state.update(left, right, dt, rearmed=rearmed)
        
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
                
        # ====== SEND CONTROL MESSAGE ======
        # Only send at configured rate
        if now - self._last_send_time >= self._send_interval:
            await self._send_control_message(hands_ok, lost_active)
            self._last_send_time = now
            
        # ====== PREVIEW DISPLAY ======
        if self.show_preview:
            self._draw_preview(frame, h, w, left, right, arming_progress, 
                             both_open, hands_ok, calib_info, lost_active)
            cv2.imshow("Hand Control Client", frame)
            
    async def _send_control_message(self, hands_ok: bool, lost_active: bool) -> None:
        """Create, validate, and send a control message."""
        # Determine if we should send enable=true
        # Drive only enabled when: user toggled enable AND frame valid AND hands ok
        can_enable = (
            self.drive_state.enabled and
            hands_ok and
            not self.drive_state.calibrating and
            self.drive_state.baseline is not None
        )
        
        # If hands lost for too long, force disable
        if lost_active:
            can_enable = False
            
        # Get velocity commands
        linear, angular = self.drive_state.get_velocity_commands(
            self.max_linear, self.max_angular
        )
        
        # If not enabled, zero the commands
        if not can_enable:
            linear = 0.0
            angular = 0.0

        # E-stop override: while latched, force motion to zero and disable, on top
        # of publishing the flag. Client-side effect — the raspi must honor estop
        # too, but we don't wait for it to act.
        if self.gesture_state.estop:
            linear = 0.0
            angular = 0.0
            can_enable = False

        # Create message
        msg = create_control_message(
            linear=linear,
            angular=angular,
            enable=can_enable,
            mode=self.gesture_state.mode,
            lane=self.gesture_state.lane,
            estop=self.gesture_state.estop,
        )
        
        # Validate message
        clamped_msg, valid, reason = self.validator.clamp_and_validate(msg)
        
        if not valid:
            logger.warning(f"Message validation failed: {reason}")
            return
            
        # Send message
        if self.ws_client and self.ws_client.connected:
            success = await self.ws_client.send_async(clamped_msg)
            if success:
                self._force_stop_sent = False
                
    async def _send_force_stop(self) -> None:
        """Send force stop message due to frame quality issues."""
        if self._force_stop_sent:
            return
            
        logger.warning("Sending force stop due to frame quality issues")
        
        # Force drive state to stop
        self.drive_state.force_stop()
        self.drive_state.enabled = False
        
        # Create and send stop message
        msg = ControlMessage.stop_message()
        
        if self.ws_client and self.ws_client.connected:
            success = await self.ws_client.send_async(msg)
            if success:
                self._force_stop_sent = True
                logger.info("Force stop message sent")
                
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
        
    async def _on_connected(self) -> None:
        """Callback when WebSocket connects."""
        logger.info("Connected to server")
        
    async def _on_disconnected(self) -> None:
        """Callback when WebSocket disconnects."""
        logger.warning("Disconnected from server")
        
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
            
        cv2.putText(frame, status, (20, 40), self.font, 0.9, color, 2)
        
        # Connection status
        conn_status = "Connected" if (self.ws_client and self.ws_client.connected) else "Disconnected"
        conn_color = (0, 255, 0) if conn_status == "Connected" else (0, 0, 255)
        cv2.putText(frame, f"Server: {conn_status}", (20, 70), self.font, 0.5, conn_color, 1)

        # Discrete-gesture state: mode / lane / e-stop
        gs = self.gesture_state
        mode_color = (0, 200, 255) if gs.mode == "onroad" else (180, 180, 180)
        cv2.putText(frame, f"Mode: {gs.mode.upper()}", (20, 100), self.font, 0.5, mode_color, 1)
        cv2.putText(frame, f"Lane: {gs.lane.upper()}", (20, 125), self.font, 0.5, mode_color, 1)
        if gs.estop:
            # Loud, centered banner — this is the one you must not miss.
            cv2.putText(frame, "*** E-STOP ***", (w // 2 - 120, 50), self.font, 1.0, (0, 0, 255), 3)
        if gs.last_event:
            cv2.putText(frame, gs.last_event, (20, 150), self.font, 0.5, (0, 255, 0), 1)
        
        # Command display
        linear, angular = self.drive_state.get_velocity_commands(self.max_linear, self.max_angular)
        cv2.putText(
            frame,
            f"Linear: {linear:+.3f} m/s",
            (20, h - 70),
            self.font, 0.7, (255, 0, 0), 2
        )
        cv2.putText(
            frame,
            f"Angular: {angular:+.3f} rad/s",
            (20, h - 40),
            self.font, 0.7, (0, 255, 255), 2
        )
        
        # Status messages
        if calib_info.get('calibrating'):
            progress = int(calib_info.get('progress', 0) * 100)
            cv2.putText(
                frame,
                f"Calibrating... {progress}%",
                (20, h - 100),
                self.font, 0.6, (0, 200, 0), 2
            )
        elif self.drive_state.calib_requested:
            cv2.putText(
                frame,
                "Hold both hands still to calibrate",
                (20, h - 100),
                self.font, 0.6, (0, 165, 255), 2
            )
        elif self.drive_state.baseline is None:
            cv2.putText(
                frame,
                "Show both hands to calibrate",
                (20, h - 100),
                self.font, 0.6, (0, 165, 255), 2
            )
        elif lost_active:
            cv2.putText(
                frame,
                "SAFE STOP (hands lost)",
                (20, h - 100),
                self.font, 0.6, (0, 0, 255), 2
            )
        elif not hands_ok:
            cv2.putText(
                frame,
                "Need both hands visible",
                (20, h - 100),
                self.font, 0.6, (0, 165, 255), 2
            )
            
        # Frame gate stats
        fg_stats = self.frame_gate.get_stats()
        if fg_stats['invalid_frames'] > 0:
            invalid_pct = fg_stats['invalid_frames'] / max(fg_stats['total_frames'], 1) * 100
            cv2.putText(
                frame,
                f"Frame errors: {invalid_pct:.1f}%",
                (w - 200, 30),
                self.font, 0.5, (0, 0, 255), 1
            )


async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point."""
    client = HandControlClient(
        server_url=args.server,
        token=args.token,
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
        "--server",
        type=str,
        default="ws://127.0.0.1:8080/control",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Authentication token",
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


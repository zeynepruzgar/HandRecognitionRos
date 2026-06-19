"""
Discrete hand-gesture commands (separate from the continuous handlebar driving).

The handlebar logic in hand_control.py is *continuous* — hand position maps to
speed/steering every frame. This module handles *discrete* events: one pose ->
one command, fired once. Keeping it separate stops the two from tangling.

Design constraints (see CLAUDE.md / chat history):
- Always-live poses: commands must be recognizable even mid-drive.
- No collision with driving: during handlebar driving both hands are OPEN, so
  every discrete pose here requires hands that are NOT open palms (thumbs, or a
  single pointing finger). That's what keeps them from misfiring.

E-stop is NOT a gesture — it's a keyboard key (handled in main.py), because a
vision-based stop fails exactly when vision fails. `estop` lives here as state
so it rides in the message and clears on re-arm, but it's set from the keyboard.
The real emergency stop is still "hands out of frame" (the visibility safety in
DriveState); this is the convenience layer on top.
"""

from typing import Optional, Dict

from .hand_control import (
    per_finger_openness,
    thumb_openness_01,
    INDEX_MCP,
    INDEX_TIP,
)

# Lane order, left to right. Stepping is clamped at the ends (no wrap-around —
# "switch left" while already in the left lane should do nothing, not jump right).
LANES = ("left", "center", "right")

MODE_ONROAD = "onroad"    # automatic with assistance (ADAS)
MODE_OFFROAD = "offroad"  # manual


# ============================================================================
# Pose detectors (pure functions on a landmark list)
# ============================================================================

def is_thumb_up(lm, fist_thresh: float, open_thresh: float) -> bool:
    """Four fingers folded + thumb sticking out (thumbs-up / thumbs-sideways)."""
    fingers = per_finger_openness(lm)
    folded = all(v <= fist_thresh for v in fingers.values())
    return folded and thumb_openness_01(lm) >= open_thresh


def pointing_direction(lm, dx_thresh: float) -> Optional[str]:
    """Index extended, other fingers folded -> 'left' / 'right' / None.

    Direction is the index vector (knuckle -> tip) in x. Landmarks come from an
    already-mirrored frame (main flips horizontally), so larger x = screen right.
    """
    fingers = per_finger_openness(lm)
    if fingers["index"] < 0.6:
        return None
    if any(fingers[f] > 0.5 for f in ("middle", "ring", "pinky")):
        return None
    dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
    if dx <= -dx_thresh:
        return "left"
    if dx >= dx_thresh:
        return "right"
    return None


# ============================================================================
# State machine
# ============================================================================

class GestureState:
    """Tracks discrete command state driven by debounced, one-shot poses."""

    def __init__(
        self,
        fist_thresh: float = 0.35,
        open_thresh: float = 0.70,
        point_dx: float = 0.06,
        trigger_hold: float = 0.30,
    ):
        # Command state (the stuff that rides in the control message).
        self.mode = MODE_OFFROAD   # safe default: manual
        self.lane = "center"
        self.estop = False         # set from the keyboard (main.py), cleared on re-arm

        # Tuning
        self.fist_thresh = fist_thresh    # openness below this = folded
        self.open_thresh = open_thresh    # thumb extension above this = "up"
        self.point_dx = point_dx          # min index x-deflection to count as a point
        self.trigger_hold = trigger_hold  # hold time before a normal pose fires

        # Per-gesture edge state: a pose must be held `hold` seconds to fire, then
        # is suppressed until released. That's what makes one pose = one command
        # instead of one-per-frame.
        self._hold: Dict[str, float] = {}
        self._fired: Dict[str, bool] = {}

        # Last thing that fired, for the preview overlay.
        self.last_event: Optional[str] = None

    def _edge(self, name: str, active: bool, dt: float, hold: float) -> bool:
        """Rising-edge detector with hold-time debounce. Fires once per press."""
        if not active:
            self._hold[name] = 0.0
            self._fired[name] = False
            return False
        self._hold[name] = self._hold.get(name, 0.0) + dt
        if self._hold[name] >= hold and not self._fired.get(name, False):
            self._fired[name] = True
            return True
        return False

    def update(self, left, right, dt: float, rearmed: bool = False) -> Dict[str, object]:
        """Advance the state machine one frame.

        Args:
            left, right: hand tuples (landmarks, label) or None, from extract_hands.
            dt: seconds since last frame.
            rearmed: True on the frame the drive was (re-)armed via the existing
                both-palms-open gesture. Re-arming clears a latched e-stop, so we
                don't need a separate "unstop" pose.
        """
        # Re-arm clears the latch. Done first so a same-frame fist can't immediately
        # re-trip it (you'd have to release and re-fist).
        if rearmed:
            self.estop = False

        l_lm = left[0].landmark if left else None
        r_lm = right[0].landmark if right else None

        # MODE toggle: both thumbs up.
        l_thumb = l_lm is not None and is_thumb_up(l_lm, self.fist_thresh, self.open_thresh)
        r_thumb = r_lm is not None and is_thumb_up(r_lm, self.fist_thresh, self.open_thresh)
        if self._edge("mode", l_thumb and r_thumb, dt, self.trigger_hold):
            self.mode = MODE_ONROAD if self.mode == MODE_OFFROAD else MODE_OFFROAD
            self.last_event = f"MODE {self.mode.upper()}"

        # LANE changes: only meaningful on-road (off-road manual has no lanes).
        if self.mode == MODE_ONROAD:
            l_left = l_lm is not None and pointing_direction(l_lm, self.point_dx) == "left"
            if self._edge("lane_left", l_left, dt, self.trigger_hold):
                self._step_lane(-1)
            r_right = r_lm is not None and pointing_direction(r_lm, self.point_dx) == "right"
            if self._edge("lane_right", r_right, dt, self.trigger_hold):
                self._step_lane(+1)
        else:
            # Keep edge state reset so flipping back to on-road doesn't insta-fire
            # a lane change from a pose that was already being held.
            self._edge("lane_left", False, dt, self.trigger_hold)
            self._edge("lane_right", False, dt, self.trigger_hold)

        return {"mode": self.mode, "lane": self.lane, "estop": self.estop}

    def _step_lane(self, delta: int) -> None:
        i = LANES.index(self.lane)
        i = max(0, min(len(LANES) - 1, i + delta))
        if LANES[i] != self.lane:
            self.lane = LANES[i]
            self.last_event = f"LANE {self.lane.upper()}"

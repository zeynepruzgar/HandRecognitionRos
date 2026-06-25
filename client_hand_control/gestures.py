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
    hand_openness_01,
    per_finger_openness,
    thumb_openness_01,
    INDEX_MCP,
    INDEX_TIP,
    THUMB_MCP,
    THUMB_TIP,
)

MODE_ONROAD = "onroad"    # automatic with assistance (ADAS) — the rover drives itself
MODE_OFFROAD = "offroad"  # manual


# ============================================================================
# Pose detectors (pure functions on a landmark list)
# ============================================================================

def is_fist(lm, fist_thresh: float, thumb_thresh: float) -> bool:
    """A real fist: all four fingers folded AND the thumb tucked.

    The thumb-tuck check is what makes a fist distinct from a thumbs-up. Without
    it, a thumbs-up (fingers folded, thumb out) also reads as a fist, because
    hand_openness_01 only averages the four fingers and ignores the thumb — so
    both gestures would cross-fire. Mutually exclusive by construction:
      fist      = fingers folded + thumb tucked  (thumb < thumb_thresh)
      thumbs-up = fingers folded + thumb out      (thumb >= thumb_thresh)
    """
    return hand_openness_01(lm) <= fist_thresh and thumb_openness_01(lm) < thumb_thresh


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


def is_thumb_up_vertical(lm, fist_thresh: float, open_thresh: float) -> bool:
    """Thumbs-UP: folded hand + thumb extended AND pointing up (vertical).

    Plain is_thumb_up is direction-agnostic, so a sideways thumb (the lane-change
    pose) also passes it and cross-fires the mode toggle. Requiring the thumb to
    be more vertical than horizontal — and pointing up (tip above knuckle; image
    y grows downward, so dy < 0) — keeps MODE (both thumbs up) cleanly separate
    from LANE (one thumb sideways).
    """
    if not is_thumb_up(lm, fist_thresh, open_thresh):
        return False
    dx = lm[THUMB_TIP].x - lm[THUMB_MCP].x
    dy = lm[THUMB_TIP].y - lm[THUMB_MCP].y
    return abs(dy) > abs(dx) and dy < 0


def is_index_up(lm, fist_thresh: float) -> bool:
    """Index finger extended and pointing UP, other fingers folded.

    Used for the MODE toggle (both index fingers up). Cleanly separable from the
    lane pose (thumb sideways over a fist -> index FOLDED) and from RUN (fist ->
    index folded), so it never cross-fires with them. The thumb is free here, so
    a natural pointing-up hand counts regardless of thumb position. Image y grows
    downward, so pointing up means tip above knuckle (dy < 0), and we require it
    to be more vertical than horizontal to reject a sideways point.
    """
    fingers = per_finger_openness(lm)
    if fingers["index"] < 0.6:
        return False
    if any(fingers[f] > fist_thresh for f in ("middle", "ring", "pinky")):
        return False
    dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
    dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y
    return abs(dy) > abs(dx) and dy < 0


def thumb_direction(lm, dx_thresh: float, fist_thresh: float, open_thresh: float) -> Optional[str]:
    """Thumb extended SIDEWAYS over a folded hand -> 'left' / 'right' / None.

    Lane-change pose: make a fist but stick the thumb out horizontally. Direction
    is the thumb vector (knuckle -> tip) in x. Landmarks come from an already-
    mirrored frame (main flips horizontally), so larger x = screen right.

    Kept distinct from MODE (both thumbs UP) by requiring the thumb to be more
    horizontal than vertical (|dx| > |dy|): a vertical thumbs-up never counts as a
    lane point, and a lane point needs only ONE hand while MODE needs both.
    """
    fingers = per_finger_openness(lm)
    # All four fingers folded (fist-like), thumb sticking out.
    if any(v > fist_thresh for v in fingers.values()):
        return None
    if thumb_openness_01(lm) < open_thresh:
        return None
    dx = lm[THUMB_TIP].x - lm[THUMB_MCP].x
    dy = lm[THUMB_TIP].y - lm[THUMB_MCP].y
    # Require a clearly HORIZONTAL thumb. A relaxed/resting hand in onroad easily
    # produces a diagonal thumb, so loosening this caused spurious CMD_A/CMD_D
    # that stuck the rover in a lane change and broke lane-following.
    if abs(dx) < abs(dy):
        return None
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
        # Default ON_ROAD to match the rover's own default (purePursuit starts in
        # ON_ROAD); otherwise a mode-toggle would flip the two out of sync.
        self.mode = MODE_ONROAD
        self.run = False           # onroad autonomy start/stop. Starts DISABLED.
        self.estop = False         # set from the keyboard (main.py), cleared on re-arm

        # Lane changes are RELATIVE events, not absolute positions: the rover only
        # knows CHANGE_LEFT/CHANGE_RIGHT. `lane_seq` increments on each new event so
        # a dropped (QoS-0) packet doesn't lose the command — the event repeats in
        # every message until the next one, and the consumer acts only on seq change.
        self.lane_change: Optional[str] = None  # "left" / "right" / None
        self.lane_seq = 0

        # Tuning
        self.fist_thresh = fist_thresh    # openness below this = folded
        self.open_thresh = open_thresh    # thumb extension above this = "up"
        self.point_dx = point_dx          # min index x-deflection to count as a point
        self.trigger_hold = trigger_hold  # hold time before a normal pose fires
        # Lane-change (thumb sideways) is more forgiving than MODE's thumbs-up: a
        # sideways thumb reads as less "extended" than a vertical one, so use a
        # lower threshold here.
        self.lane_thumb_open = 0.40
        self.lane_dx = 0.03               # min thumb x-deflection for a lane point

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

        Returns a dict that includes "commands": the list of raw string commands
        the rover understands (MODE_TOGGLE / CMD_W / CMD_S / CMD_A / CMD_D) that
        fired this frame. main.py forwards these over UDP. They're one-shot per
        pose, matching the rover's latching behaviour: a command keeps the rover
        doing its thing until the next command arrives.
        """
        # Re-arm clears the latch. Done first so a same-frame fist can't immediately
        # re-trip it (you'd have to release and re-fist).
        if rearmed:
            self.estop = False

        commands: list = []

        l_lm = left[0].landmark if left else None
        r_lm = right[0].landmark if right else None

        # LANE poses first (single hand, sideways thumb). Either hand can signal;
        # direction (not which hand) decides, which removes the "wrong hand" failure
        # and the mirrored-frame ambiguity. We compute these up front so the mode
        # gesture can be suppressed whenever a lane pose is present.
        l_dir = thumb_direction(l_lm, self.lane_dx, self.fist_thresh, self.lane_thumb_open) if l_lm is not None else None
        r_dir = thumb_direction(r_lm, self.lane_dx, self.fist_thresh, self.lane_thumb_open) if r_lm is not None else None
        point_left = (l_dir == "left") or (r_dir == "left")
        point_right = (l_dir == "right") or (r_dir == "right")
        lane_active = point_left or point_right

        # MODE toggle: both INDEX fingers up. Cleanly distinct from the lane pose
        # (thumb sideways over a fist -> index folded) and from RUN (fist -> index
        # folded), so no cross-fire and no need to lock it out against lane.
        l_idx = l_lm is not None and is_index_up(l_lm, self.fist_thresh)
        r_idx = r_lm is not None and is_index_up(r_lm, self.fist_thresh)
        if self._edge("mode", l_idx and r_idx, dt, self.trigger_hold):
            self.mode = MODE_ONROAD if self.mode == MODE_OFFROAD else MODE_OFFROAD
            self.last_event = f"MODE {self.mode.upper()}"
            commands.append("MODE_TOGGLE")

        # START/STOP autonomy: both fists. Distinct from open palms (arming),
        # thumbs-up (mode), and pointing (lane), so no cross-fire. The rover
        # latches on CMD_W (keep going) and stops on CMD_S.
        l_fist = l_lm is not None and is_fist(l_lm, self.fist_thresh, self.open_thresh)
        r_fist = r_lm is not None and is_fist(r_lm, self.fist_thresh, self.open_thresh)
        if self._edge("run", l_fist and r_fist, dt, self.trigger_hold):
            self.run = not self.run
            self.last_event = f"RUN {'ON' if self.run else 'OFF'}"
            commands.append("CMD_W" if self.run else "CMD_S")

        # LANE change events. The rover only acts on these in ON_ROAD while moving,
        # so we let it be the authority and don't gate on our local mode here — that
        # way a dropped MODE_TOGGLE packet can't desync us into swallowing them.
        if self._edge("lane_left", point_left, dt, self.trigger_hold):
            self._emit_lane_change("left")
            commands.append("CMD_A")
        if self._edge("lane_right", point_right, dt, self.trigger_hold):
            self._emit_lane_change("right")
            commands.append("CMD_D")

        return {
            "mode": self.mode,
            "run": self.run,
            "lane_change": self.lane_change,
            "lane_seq": self.lane_seq,
            "estop": self.estop,
            "commands": commands,
        }

    def _emit_lane_change(self, direction: str) -> None:
        self.lane_change = direction
        self.lane_seq += 1
        self.last_event = f"LANE {direction.upper()}"

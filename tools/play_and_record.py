"""
Minimal keyboard-driven recorder for human-in-the-loop LfD data collection.

Usage:
    Activate your project's venv (must have opencv-python installed) and run:
        python3 tools/play_and_record.py --outdir demonstrations

Controls (no HUD):
    W / Up Arrow    : increase speed
    S / Down Arrow  : decrease speed
    A / Left Arrow  : rotate left
    D / Right Arrow : rotate right
    SPACE           : stop (speed=0)
    R               : reset & save current episode (if any) and start a new one
    Q / ESC         : quit and save current episode

Notes:
- This script uses the environment's fast OpenCV renderer (simulator/prisoner_env.py fast render).
- No HUD overlay is drawn; the renderer window is used for visuals and cv2.waitKey to collect keystrokes.
- Saved files: per-episode compressed .npz containing lists of states and observations produced by the environment.
"""

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import math
import json
from pathlib import Path

# import the environment and a simple blue heuristic policy
from simulator.prisoner_env import PrisonerBothEnv
from blue_bc.heuristic import BlueHeuristic


def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def save_episode(outdir, episode_idx, records):
    """Save a single episode as compressed npz."""
    if len(records) == 0:
        print("No frames to save for episode", episode_idx)
        return
    fname = os.path.join(outdir, f"episode_{episode_idx:04d}.npz")
    # records is a list of dicts with identical keys
    keys = list(records[0].keys())
    arrays = {k: np.array([r[k] for r in records]) for k in keys}
    np.savez_compressed(fname, **arrays)
    print(f"Saved episode {episode_idx} -> {fname} (len={len(records)})")


def run(outdir, speed_init=12.0, theta_init=0.0, speed_step=1.0, theta_step=0.2, debug=False):
    make_dirs(outdir)

    env = PrisonerBothEnv()
    blue_policy = BlueHeuristic(env)
    blue_policy.init_behavior()

    episode_idx = 0
    records = []

    speed = float(speed_init)
    theta = float(theta_init)
    control_mode = 'direct'  # 'direct' maps WASD to world directions; 'tank' uses speed+theta

    paused = False
    print("Controls: WASD/arrows to drive, SPACE stop, R reset+save, Q quit+save")
    # keep a persistent trail of prisoner locations (in image coords) so the player
    # can see where they've been. This trail is persistent across the session and
    # is purely visual; the full trajectory is already saved in records per timestep.
    prisoner_trail = []

    # Key-hold emulation: OpenCV returns individual keycodes but not key-release
    # events. To support multi-directional presses (W+D, diagonals, etc.) we
    # maintain a timestamp of the last time each logical key was seen. Keys are
    # considered "active" if they've been pressed within `hold_timeout` seconds.
    last_key_times = {}
    hold_timeout = 0.5  # seconds; tune to feel of key-hold/auto-repeat
    # speed scaling applied when sending action to the environment to make
    # keyboard-driven movement feel snappier. You can lower this if it's too fast.
    speed_scale = 2.0

    try:
        obs = env.reset()
        prev_red_action = None
        # Arrow key autodetection support: try to load saved keycodes from
        # ~/.diffusionrl_keycodes.json. If --autocapture is used, the script
        # will prompt the user to press each arrow key and save the observed
        # codes to that file for future runs.
        keycodes_file = Path.home() / '.diffusionrl_keycodes.json'
        saved_keycodes = {}
        if keycodes_file.exists():
            try:
                saved_keycodes = json.loads(keycodes_file.read_text())
                # normalize saved_keycodes so each entry is a list of int candidates
                for k, v in list(saved_keycodes.items()):
                    if isinstance(v, list):
                        saved_keycodes[k] = [int(x) for x in v]
                    else:
                        # single value -> expand into candidate encodings
                        raw = int(v)
                        cand = [raw, raw & 0xFF, (raw >> 16) & 0xFFFF]
                        # dedupe
                        saved_keycodes[k] = sorted(set(cand))
                if debug:
                    print('Loaded saved keycodes from', keycodes_file)
            except Exception:
                saved_keycodes = {}
        while True:
            # If autocapture requested, perform capture before main loop starts
            if args_autocapture := getattr(run, '_autocapture_flag', False):
                # perform interactive capture once, populate saved_keycodes
                def capture_prompt(name):
                    print(f"Please focus the small window and press the {name} arrow key now (ESC to skip)")
                    cv2.namedWindow('KeyCapture', cv2.WINDOW_NORMAL)
                    cv2.imshow('KeyCapture', np.zeros((100, 300, 3), dtype=np.uint8))
                    k = cv2.waitKey(0)
                    cv2.destroyWindow('KeyCapture')
                    print('Captured raw key:', k, 'ascii:', k & 0xFF)
                    return int(k)

                try:
                    up_k = capture_prompt('UP')
                    down_k = capture_prompt('DOWN')
                    left_k = capture_prompt('LEFT')
                    right_k = capture_prompt('RIGHT')
                    def make_candidates(raw):
                        raw = int(raw)
                        return sorted(set([raw, raw & 0xFF, (raw >> 16) & 0xFFFF]))

                    saved_keycodes = {
                        'up': make_candidates(up_k),
                        'down': make_candidates(down_k),
                        'left': make_candidates(left_k),
                        'right': make_candidates(right_k),
                    }
                    try:
                        keycodes_file.write_text(json.dumps(saved_keycodes))
                        print('Saved keycodes to', keycodes_file)
                    except Exception as e:
                        print('Warning: failed to save keycodes:', e)
                finally:
                    # clear the autocapture flag so we don't capture again
                    run._autocapture_flag = False
            # render (fast) but don't let the env call imshow — we will overlay
            # a small HUD and present the image ourselves so key handling is
            # entirely controlled here.
            img = env.render('human', show=False, fast=True)

            # draw a minimal HUD on the image so the player can see speed/theta and mode
            try:
                hud_text = f"mode={control_mode} spd={speed:.1f} theta={theta:.2f}"
                cv2.putText(img, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # highlight prisoner location to make it easier to spot
                pr = env.get_prisoner_location()
                # fast_render_canvas flips and scales internally; draw a small red circle at approximate location
                # approximate mapping: world coords -> image coords via scale used in fast_render_canvas (default 3)
                scale = 3
                img_h, img_w = img.shape[:2]
                # convert world location to canvas coordinates (note fast_render flips vertically)
                cx = int(pr[0] / (env.dim_x / img_w))
                cy = int((env.dim_y - pr[1]) / (env.dim_y / img_h))
                # append to persistent trail
                prisoner_trail.append((cx, cy))

                # draw trail with a fading intensity (older points dimmer)
                if len(prisoner_trail) > 1:
                    for i, (tx, ty) in enumerate(prisoner_trail):
                        # intensity scales from 50 (old) to 255 (new)
                        intensity = int(50 + 205 * (i / max(1, len(prisoner_trail) - 1)))
                        cv2.circle(img, (tx, ty), 2, (0, 0, intensity), -1)

                # draw the current prisoner marker on top (slightly larger)
                cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
            except Exception:
                pass

            # Present image in a named window we control so OS-level focus works better
            cv2.namedWindow("DiffusionRL", cv2.WINDOW_NORMAL)
            cv2.imshow("DiffusionRL", img)

            # read key. cv2.waitKey may return large codes for special keys (arrow keys) or ASCII for letters.
            # It returns -1 when no key pressed.
            key = cv2.waitKey(1)

            if key != -1:
                # map keys. Support both lowercase and uppercase letters as well
                # as several common arrow-key codes OpenCV uses on different OSes.
                ascii_key = key & 0xFF
                if debug:
                    print(f"key raw={key} ascii={ascii_key}")
                # common OpenCV arrow-key codes (platform dependent). We'll
                # build these sets from saved_keycodes if available; otherwise
                # fall back to a broad default list. key_matches_arrow checks a
                # few slices of the raw key to match common encodings.
                if saved_keycodes:
                    ARROW_UP = set(int(x) for x in saved_keycodes.get('up', []))
                    ARROW_DOWN = set(int(x) for x in saved_keycodes.get('down', []))
                    ARROW_LEFT = set(int(x) for x in saved_keycodes.get('left', []))
                    ARROW_RIGHT = set(int(x) for x in saved_keycodes.get('right', []))
                else:
                    ARROW_UP = {82, 2490368, 65362}
                    ARROW_DOWN = {84, 2621440, 65364}
                    ARROW_LEFT = {81, 2424832, 65361}
                    ARROW_RIGHT = {83, 2555904, 65363}

                def key_matches_arrow(raw_key, arrow_set):
                    # check raw key, low-8 bits, and high-16 bits
                    if raw_key in arrow_set:
                        return True
                    if (raw_key & 0xFF) in arrow_set:
                        return True
                    if ((raw_key >> 16) & 0xFFFF) in arrow_set:
                        return True
                    return False

                now = time.time()

                # helper: mark a logical key active at current timestamp
                def mark_key_active(token):
                    last_key_times[token] = now

                if ascii_key in (ord('q'), ord('Q')) or key == 27:  # q or ESC
                    print('Quit received')
                    if len(records) > 0:
                        save_episode(outdir, episode_idx, records)
                    break
                elif key in (ord('r'),):
                    print('Reset and save (if any)')
                    if len(records) > 0:
                        save_episode(outdir, episode_idx, records)
                        episode_idx += 1
                    records = []
                    obs = env.reset()
                    blue_policy.reset()
                    blue_policy.init_behavior()
                    speed = speed_init
                    theta = theta_init
                    continue
                elif ascii_key in (ord('m'), ord('M')):
                    # toggle control mode
                    control_mode = 'tank' if control_mode == 'direct' else 'direct'
                    print('Control mode ->', control_mode)
                elif ascii_key == ord(' '):  # space
                    speed = 0.0
                elif ascii_key == ord('p'):
                    paused = not paused
                    print('Paused' if paused else 'Unpaused')
                else:
                    # update key timestamps for held-key emulation
                    if ascii_key in (ord('w'), ord('W')) or key_matches_arrow(key, ARROW_UP):
                        mark_key_active('w')
                        mark_key_active('up')
                    if ascii_key in (ord('s'), ord('S')) or key_matches_arrow(key, ARROW_DOWN):
                        mark_key_active('s')
                        mark_key_active('down')
                    if ascii_key in (ord('a'), ord('A')) or key_matches_arrow(key, ARROW_LEFT):
                        mark_key_active('a')
                        mark_key_active('left')
                    if ascii_key in (ord('d'), ord('D')) or key_matches_arrow(key, ARROW_RIGHT):
                        mark_key_active('d')
                        mark_key_active('right')

                    # compute active keys (within hold_timeout)
                    active = {k for k, t in last_key_times.items() if now - t < hold_timeout}

                    # WASD/arrow controls — behavior depends on control_mode
                    if control_mode == 'tank':
                        # W/S increase/decrease speed
                        if 'w' in active or 'up' in active:
                            speed = float(min(80.0, speed + speed_step))
                        if 's' in active or 'down' in active:
                            speed = float(max(0.0, speed - speed_step))
                        # A/D rotate (can be combined with W/S)
                        if 'a' in active or 'left' in active:
                            theta -= theta_step
                        if 'd' in active or 'right' in active:
                            theta += theta_step
                    else:
                        # direct mode: combine active direction keys for diagonal movement
                        direct_dx = 0
                        direct_dy = 0
                        if 'w' in active or 'up' in active:
                            direct_dy += 1
                        if 's' in active or 'down' in active:
                            direct_dy -= 1
                        if 'a' in active or 'left' in active:
                            direct_dx -= 1
                        if 'd' in active or 'right' in active:
                            direct_dx += 1

                        if direct_dx != 0 or direct_dy != 0:
                            # compute heading (dy positive -> up)
                            theta = float(math.atan2(direct_dy, direct_dx))
                            # responsive default speed when starting from near-zero
                            if speed < 1.0:
                                speed = 32.0

            if paused:
                time.sleep(0.02)
                continue

            # Build red action depending on control_mode. In 'direct' mode we map
            # WASD to immediate direction; in 'tank' mode we treat speed+theta as
            # differential controls. Apply speed scaling so keyboard-driven motion
            # is more responsive; the scaling is internal to the recorder only.
            red_action = np.array([float(speed * speed_scale), theta], dtype=np.float32)
            if debug:
                if prev_red_action is None or not np.allclose(prev_red_action, red_action):
                    print('red_action ->', red_action)
                prev_red_action = red_action.copy()

            # blue action from heuristic
            blue_action = blue_policy.get_each_action()

            # step the combined environment
            red_obs, blue_obs, reward, done, blue_detect_idx, is_hs_detected = env.step_both(red_action, blue_action)

            # record useful info
            sp_locs, heli_locs = env.get_blue_locations()
            try:
                blue_locs = np.array(sp_locs + heli_locs)
            except Exception:
                # fallback to object array if shapes are somehow inconsistent
                blue_locs = np.array([sp_locs, heli_locs], dtype=object)

            rec = {
                'timestep': env.timesteps,
                'red_action': red_action.copy(),
                'red_obs': red_obs.copy() if hasattr(red_obs, 'copy') else np.array(red_obs),
                'blue_obs': np.array(blue_obs) if blue_obs is not None else np.array([]),
                'reward': float(reward),
                'done': bool(done),
                'prisoner_location': np.array(env.get_prisoner_location()),
                'blue_locations': blue_locs,
            }
            records.append(rec)

            if done:
                print('Episode done at timestep', env.timesteps)
                save_episode(outdir, episode_idx, records)
                episode_idx += 1
                records = []
                obs = env.reset()
                blue_policy.reset()
                blue_policy.init_behavior()
                speed = speed_init
                theta = theta_init

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='demos', help='directory to save episodes (.npz)')
    parser.add_argument('--speed', type=float, default=3.0, help='initial speed')
    parser.add_argument('--theta', type=float, default=0.0, help='initial heading in radians')
    parser.add_argument('--debug', action='store_true', help='enable debug prints for keycodes and actions')
    parser.add_argument('--autocapture', action='store_true', help='interactively capture arrow keycodes and save to ~/.diffusionrl_keycodes.json')
    args = parser.parse_args()

    # If user requested autocapture, set a temporary flag on the run function
    # that will trigger the capture prompt on first entry into the loop.
    if args.autocapture:
        setattr(run, '_autocapture_flag', True)

    run(args.outdir, speed_init=args.speed, theta_init=args.theta, debug=args.debug)

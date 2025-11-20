"""
Minimal keyboard-driven recorder for human-in-the-loop LfD data collection.

Usage:
    Activate your project's venv (must have opencv-python installed) and run:
        python3 tools/play_and_record.py --outdir demonstrations

Controls:
    W / Up Arrow    : move up
    S / Down Arrow  : move down
    A / Left Arrow  : move left
    D / Right Arrow : move right
    SPACE           : stop (speed=0)
    R               : reset & save current episode (if any) and start a new one
    Q / ESC         : quit and save current episode

Notes:
- This script uses the environment's fast OpenCV renderer (simulator/prisoner_env.py fast render).

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


def run(outdir, speed_init=5.0, theta_init=0.0, seed=None):
    make_dirs(outdir)
    
    # Set random seed if provided for reproducible camera/hideout/spawn placement
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    # Create environment with randomized cameras, hideout locations, and spawn position
    # camera_net_bool=False disables the camera net cluster in top-right corner
    # spawn_mode='uniform' randomizes starting position across the map
    # Use pre-generated Perlin noise terrain maps for natural forest patterns
    env = PrisonerBothEnv(
        terrain_map='simulator/forest_coverage/map_set',
        random_cameras=True,
        random_hideout_locations=True,
        camera_net_bool=False,
        num_random_known_cameras=5,
        num_random_unknown_cameras=25,
        spawn_mode='uniform'
    )
    blue_policy = BlueHeuristic(env, debug=False)  # Set debug=True to see blue team planning (saves plots to logs/temp/)
    blue_policy.init_behavior()

    episode_idx = 0
    records = []

    speed = float(speed_init)
    theta = float(theta_init)

    paused = False
    print("Controls: WASD/arrows to drive, SPACE stop, R reset+save, Q quit+save")
    # keep a persistent trail of prisoner locations (in image coords) so the player
    # can see where they've been. This trail is persistent across the session and
    # is purely visual; the full trajectory is already saved in records per timestep.
    prisoner_trail = []

    # Key-hold tracking: OpenCV returns individual keycodes but not key-release
    # events. We track which frame each key was last seen on. Keys stay active
    # for enough frames to bridge OS auto-repeat (~250ms), balancing hold vs release responsiveness.
    last_key_frame = {}  # maps key name -> frame number when last seen
    frame_counter = 0
    key_hold_frames = 10  # keys stay active for 10 frames (~167ms) to allow multi-key detection
    
    # Smooth movement parameters
    max_speed = 7.5  # match fugitive speed limit for realistic detection
    acceleration = 100.0  # instant acceleration to max speed
    deceleration = 100.0  # instant deceleration to stop

    try:
        obs = env.reset()
        red_obs = obs  # Initial observation for red (prisoner)
        blue_obs = np.zeros(env.blue_observation_space.shape)  # Initialize blue obs
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
            except Exception:
                saved_keycodes = {}
        while True:
            frame_counter += 1
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
            try:
                img = env.render('human', show=False, fast=True)
                
                # Make a copy to avoid any potential corruption
                img = img.copy()
            except Exception as e:
                print(f"ERROR: Rendering failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # draw a minimal HUD on the image so the player can see speed/theta
            try:
                # Show detection status and active keys for debugging
                detection_status = "DETECTED!" if env.is_detected else "Hidden"
                active_str = ','.join(sorted(active_keys)) if active_keys else "none"
                
                # Find nearest camera and show distance
                prisoner_loc = np.array(env.get_prisoner_location())
                min_cam_dist = float('inf')
                for cam in env.camera_list:
                    dist = np.linalg.norm(prisoner_loc - np.array(cam.location))
                    if dist < min_cam_dist:
                        min_cam_dist = dist
                
                # Show detection range for current speed
                if speed == 0:
                    detect_range = 1.0
                else:
                    # Approximate: detection_factor(4.0) * terrain(~1.0) * type(1.0) * speed + 1
                    detect_range = 4.0 * speed + 1
                
                hud_text = f"spd={speed:.1f} | {detection_status} | keys={active_str} | nearest_cam={min_cam_dist:.1f} | detect_range={detect_range:.0f}"
                cv2.putText(img, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
            except Exception as e:
                # If rendering fails, at least show the error so we know what's wrong
                print(f"Warning: HUD rendering failed: {e}")

            # Present image in a named window we control so OS-level focus works better
            cv2.namedWindow("DiffusionRL", cv2.WINDOW_NORMAL)
            cv2.imshow("DiffusionRL", img)

            # Check if window was closed by user
            if cv2.getWindowProperty("DiffusionRL", cv2.WND_PROP_VISIBLE) < 1:
                print('Window closed by user')
                if len(records) > 0:
                    save_episode(outdir, episode_idx, records)
                break

            # Read key. cv2.waitKey may return large codes for special keys (arrow keys) or ASCII for letters.
            # It returns -1 when no key pressed.
            key = cv2.waitKey(1)

            if key != -1:
                # map keys. Support both lowercase and uppercase letters as well
                # as several common arrow-key codes OpenCV uses on different OSes.
                ascii_key = key & 0xFF
                # Print raw keycode and ASCII low-8bits to help debug
                # platform-dependent arrow encodings. This prints for every
                # key event so you can see exactly what the OS/OpenCV is
                # returning when you press a key.
                # print(f"Key event: raw={key} ascii={ascii_key}")  # commented out - may interfere with rendering
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
                    # Common OpenCV/platform-dependent arrow key codes. Keep
                    # this list broad to maximize chance of matching across
                    # macOS, Linux, and Windows builds of OpenCV.
                    ARROW_UP = {82, 2490368, 65362, 63232}
                    ARROW_DOWN = {84, 2621440, 65364, 63233}
                    ARROW_LEFT = {81, 2424832, 65361, 63234}
                    ARROW_RIGHT = {83, 2555904, 65363, 63235}

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
                # collect a list of logical tokens that this keypress corresponds to
                pressed_tokens = []
                # letters
                if ascii_key in (ord('q'), ord('Q')) or key == 27:
                    pressed_tokens.append('QUIT')
                if ascii_key in (ord('r'),):
                    pressed_tokens.append('RESET')
                if ascii_key in (ord('m'), ord('M')):
                    pressed_tokens.append('MODE_TOGGLE')
                if ascii_key == ord(' '):
                    pressed_tokens.append('SPACE')
                if ascii_key == ord('p'):
                    pressed_tokens.append('PAUSE')
                # WASD
                if ascii_key in (ord('w'), ord('W')):
                    pressed_tokens.append('w')
                if ascii_key in (ord('s'), ord('S')):
                    pressed_tokens.append('s')
                if ascii_key in (ord('a'), ord('A')):
                    pressed_tokens.append('a')
                if ascii_key in (ord('d'), ord('D')):
                    pressed_tokens.append('d')
                # arrows (check various encodings)
                if saved_keycodes:
                    if key_matches_arrow(key, ARROW_UP):
                        pressed_tokens.append('up')
                    if key_matches_arrow(key, ARROW_DOWN):
                        pressed_tokens.append('down')
                    if key_matches_arrow(key, ARROW_LEFT):
                        pressed_tokens.append('left')
                    if key_matches_arrow(key, ARROW_RIGHT):
                        pressed_tokens.append('right')
                else:
                    # still check against fallback sets
                    if key_matches_arrow(key, ARROW_UP):
                        pressed_tokens.append('up')
                    if key_matches_arrow(key, ARROW_DOWN):
                        pressed_tokens.append('down')
                    if key_matches_arrow(key, ARROW_LEFT):
                        pressed_tokens.append('left')
                    if key_matches_arrow(key, ARROW_RIGHT):
                        pressed_tokens.append('right')

                if pressed_tokens:
                    print(f'Frame {frame_counter}: Detected tokens:', pressed_tokens)

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
                    red_obs = obs
                    blue_obs = np.zeros(env.blue_observation_space.shape)
                    blue_policy.reset()
                    blue_policy.init_behavior()
                    speed = speed_init
                    theta = theta_init
                    continue
                elif ascii_key == ord(' '):  # space
                    # Clear all directional keys for immediate stop
                    last_key_frame.clear()
                    speed = 0.0
                elif ascii_key == ord('p'):
                    paused = not paused
                    print('Paused' if paused else 'Unpaused')
                else:
                    # Add currently pressed directional keys to active set
                    if ascii_key in (ord('w'), ord('W')) or key_matches_arrow(key, ARROW_UP):
                        last_key_frame['w'] = frame_counter
                        last_key_frame['up'] = frame_counter
                    if ascii_key in (ord('s'), ord('S')) or key_matches_arrow(key, ARROW_DOWN):
                        last_key_frame['s'] = frame_counter
                        last_key_frame['down'] = frame_counter
                    if ascii_key in (ord('a'), ord('A')) or key_matches_arrow(key, ARROW_LEFT):
                        last_key_frame['a'] = frame_counter
                        last_key_frame['left'] = frame_counter
                    if ascii_key in (ord('d'), ord('D')) or key_matches_arrow(key, ARROW_RIGHT):
                        last_key_frame['d'] = frame_counter
                        last_key_frame['right'] = frame_counter

            # Compute which keys are currently active (seen within last few frames)
            active_keys = {k for k, f in last_key_frame.items() if frame_counter - f <= key_hold_frames}

            # Debug output: show current state
            prisoner_loc = env.get_prisoner_location()
            sp_locs, heli_locs = env.get_blue_locations()
            print(f'Frame {frame_counter}: Prisoner@({prisoner_loc[0]:.1f},{prisoner_loc[1]:.1f}) | '
                  f'Active keys: {sorted(active_keys) if active_keys else "none"} | '
                  f'Speed: {speed:.1f} | Theta: {theta:.2f} | '
                  f'Detected: {"YES" if env.is_detected else "no"} | '
                  f'Search parties: {len(sp_locs)} | Helicopters: {len(heli_locs)}')

            # Direct control: combine active direction keys for diagonal movement
            direct_dx = 0
            direct_dy = 0
            if 'w' in active_keys or 'up' in active_keys:
                direct_dy += 1
            if 's' in active_keys or 'down' in active_keys:
                direct_dy -= 1
            if 'a' in active_keys or 'left' in active_keys:
                direct_dx -= 1
            if 'd' in active_keys or 'right' in active_keys:
                direct_dx += 1

            if direct_dx != 0 or direct_dy != 0:
                # compute target heading (dy positive -> up)
                target_theta = float(math.atan2(direct_dy, direct_dx))
                
                # Calculate shortest angular distance
                angle_diff = target_theta - theta
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Fast but smooth turning for responsive diagonal transitions
                if abs(angle_diff) < 0.1:
                    theta = target_theta  # Snap when very close
                else:
                    turn_rate = 0.3  # Fast turning rate for video game feel
                    theta += angle_diff * turn_rate
                
                # Instant max speed
                speed = max_speed
            else:
                # no keys pressed - smooth deceleration
                if speed > 0:
                    speed = max(speed - deceleration, 0.0)
                else:
                    speed = 0.0

            if paused:
                time.sleep(0.02)
                continue

            # Build red action from current speed and heading.
            # WASD/arrows map to immediate direction and stop when released.
            red_action = np.array([float(speed), theta], dtype=np.float32)

            # blue action from heuristic - pass blue_obs so it can detect the prisoner
            blue_action = blue_policy.predict(blue_obs)

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
                # Show why the episode ended
                if env.timesteps >= env.max_timesteps:
                    print(f'Episode done: Time limit reached ({env.timesteps}/{env.max_timesteps} timesteps)')
                elif env.near_goal:
                    print(f'Episode done: Reached hideout at timestep {env.timesteps}')
                else:
                    print(f'Episode done at timestep {env.timesteps}')
                save_episode(outdir, episode_idx, records)
                episode_idx += 1
                records = []
                
                # Reset environment for next episode
                obs = env.reset()
                red_obs = obs
                blue_obs = np.zeros(env.blue_observation_space.shape)
                speed = speed_init
                theta = theta_init

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='demos', help='directory to save episodes (.npz)')
    parser.add_argument('--speed', type=float, default=5.0, help='initial speed')
    parser.add_argument('--theta', type=float, default=0.0, help='initial heading in radians')
    parser.add_argument('--autocapture', action='store_true', help='interactively capture arrow keycodes and save to ~/.diffusionrl_keycodes.json')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducible camera/hideout/spawn placement')
    args = parser.parse_args()

    # If user requested autocapture, set a temporary flag on the run function
    # that will trigger the capture prompt on first entry into the loop.
    if args.autocapture:
        setattr(run, '_autocapture_flag', True)

    run(args.outdir, speed_init=args.speed, theta_init=args.theta, seed=args.seed)

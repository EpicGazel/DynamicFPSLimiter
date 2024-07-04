import time
import mss
import numpy as np
import cv2
import pygetwindow as gw
from pynput.keyboard import Key, Controller, KeyCode
import imagehash
from PIL import Image

# Constants
BASE_SIMILARITY_THRESHOLD = 0.25  # 0.90  # Initial similarity threshold
SIMILARITY_DURATION = 1  # Seconds of similarity required to switch to low limit
GRACE_PERIOD = 13  # Seconds to stay at high limit before switching back
FRAME_HISTORY_SIZE = 27  # Number of frames to include in the rolling average
FRAMES_FOR_SWITCH_HIGH = 4  # Smaller amount of frames that must be different to switch to high limit

HIGH_FPS_KEY = KeyCode.from_vk(105)
LOW_FPS_KEY = KeyCode.from_vk(104)
WINDOW_NAME = 'VRChat'


# Bigger = less similar, lower = more similar, 0 = very similar
def calculate_similarity(frame1, frame2):
    # https://stackoverflow.com/questions/65440298/quick-technique-for-comparing-images-better-than-mse-in-python
    score = imagehash.average_hash(Image.fromarray(frame1)) - imagehash.average_hash(Image.fromarray(frame2))
    return score


keyboard = Controller()


def set_framerate_limit(low_limit=True):
    if low_limit:
        print("Setting framerate limit to low (ctrl + alt + num 8)")
        with keyboard.pressed(Key.ctrl):  # ctrl+ alt + num 8
            with keyboard.pressed(Key.alt):
                keyboard.press(LOW_FPS_KEY)
                keyboard.release(LOW_FPS_KEY)
    else:
        print("Setting framerate limit to high (ctrl + alt + num 9)")
        with keyboard.pressed(Key.ctrl):  # ctrl + alt + num 9
            with keyboard.pressed(Key.alt):
                keyboard.press(HIGH_FPS_KEY)
                keyboard.release(HIGH_FPS_KEY)


def get_window():
    for window in gw.getWindowsWithTitle(WINDOW_NAME):
        if WINDOW_NAME in window.title:
            return window
    return None


def main():
    with mss.mss() as sct:
        prev_frame = None
        low_limit = False
        grace_timer = 0
        frame_similarity_scores = []
        adaptive_threshold = BASE_SIMILARITY_THRESHOLD  # TODO: Not implemented yet
        debug_print_timer = 0
        sct.compression_level = 0  # Doesn't seem to have an effect

        while True:
            start_time = time.time()

            window = get_window()
            if window:
                left, top, right, bottom = window.left, window.top, window.right, window.bottom
                window_rect = {'top': top, 'left': left, 'width': right - left, 'height': bottom - top}

                current_frame = np.array(sct.grab(window_rect))

                height, width, _ = current_frame.shape
                new_width = int(width * 0.2)
                new_height = int(height * 0.2)
                current_frame = cv2.resize(current_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                # cv2.INTER_AREA took 17.2% while INTER_NEAREST took 1%
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is None:
                    prev_frame = current_frame
                    continue

                similarity_score = calculate_similarity(prev_frame, current_frame)

                frame_similarity_scores.append(similarity_score)
                if len(frame_similarity_scores) > FRAME_HISTORY_SIZE:
                    frame_similarity_scores.pop(0)
                avg_similarity_all = np.mean(frame_similarity_scores)
                avg_similarity_small = np.mean(frame_similarity_scores[-FRAMES_FOR_SWITCH_HIGH:])

                if debug_print_timer >= 10:
                    print(f"[{time.strftime('%H:%M:%S')}]")
                    print(f"\tFrame Similarity (SSIM): {similarity_score}")
                    print(f"\tAvg Similarity for past {FRAME_HISTORY_SIZE} frames: {avg_similarity_all}")
                    print(f"\tAvg Similarity for past {FRAMES_FOR_SWITCH_HIGH} frames: {avg_similarity_small}")
                    debug_print_timer = 0
                else:
                    debug_print_timer += 1

                # Switch to low limit if similarity is above threshold (very similar) and grace period is over
                if avg_similarity_all < adaptive_threshold and time.time() > grace_timer:
                    if not low_limit:  # and not already in low_limit
                        print(
                            f"[{time.strftime('%H:%M:%S')}] Switching to LOW framerate limit after {FRAME_HISTORY_SIZE} frames of similarity.")  # after {SIMILARITY_DURATION} seconds of similarity.")
                        set_framerate_limit(low_limit=True)
                        low_limit = True
                        # adaptive_threshold = BASE_SIMILARITY_THRESHOLD  # Reset threshold
                elif avg_similarity_small > adaptive_threshold and low_limit:
                    print(f"[{time.strftime('%H:%M:%S')}] Frames are distinct. Switching to HIGH framerate limit.")
                    set_framerate_limit(low_limit=False)
                    low_limit = False
                    grace_timer = time.time() + GRACE_PERIOD

                prev_frame = current_frame

            else:
                if grace_timer != 0:
                    print(f"[{time.strftime('%H:%M:%S')}] VRChat window not found. Resetting timers.")
                grace_timer = 0

            # sleep for the time remaining in the current frame at a rate of 24 fps
            time_elapsed = time.time() - start_time
            time_remaining = 1 / 24 - time_elapsed
            if time_remaining > 0:
                time.sleep(time_remaining)
            else:
                print(f"Calculations took longer than 1/24th of a second. Time elapsed: {time_elapsed} seconds.")


if __name__ == "__main__":
    main()

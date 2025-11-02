import sys
sys.path.append("/liujinxin/code/lhc/lerobot/")
sys.path.append("/liujinxin/code/lhc/lerobot/LIBERO/")
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
import time
import imageio
import logging
import math
import numpy as np
import base64
import io

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_libero_env(task, resolution=256):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  
    return env, task_description

def get_libero_dummy_action():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1] 

def get_libero_image(obs):
    img = obs["agentview_image"]
    img = img[::-1, ::-1] 
    return img

def get_libero_wrist_image(obs):
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1] 
    return img

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir=None):
    if output_dir is None:
        DATE = time.strftime("%Y_%m_%d")
        rollout_dir = f"./rollouts/{DATE}"
    else:
        rollout_dir = output_dir
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--unified--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    log_message(f"保存回放视频到路径 {mp4_path}", log_file)
    return mp4_path



def ndarray_image_to_base64(img: np.ndarray, format: str = "JPG") -> str:
    if img.dtype != np.uint8:
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        img_uint8 = arr.astype(np.uint8)
    else:
        img_uint8 = img

    buffer = io.BytesIO()
    imageio.v2.imwrite(buffer, img_uint8, format=format)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return b64



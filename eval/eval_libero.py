import os
# os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
import sys
sys.path.append("/liujinxin/zhy/lirunze/vla-0/models")
sys.path.append("/liujinxin/code/lhc/lerobot/")
sys.path.append("/liujinxin/code/lhc/lerobot/LIBERO/")
from libero.libero import benchmark
from Qwen2_5_VL_3B_VLA import Qwen2_5_VL_3B_VLA
from PIL import Image
from dataclasses import dataclass
from enum import Enum
from collections import deque
import tqdm
import argparse
from libero_utils import *


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# 每次预测的动作步数
NUM_ACTIONS_CHUNK = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class EvaluationConfig:
    # 模型相关参数
    model_path: str = ""
    action_norm_stats_path: str = ""
    
    # LIBERO环境参数
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    num_open_loop_steps: int = 8
    resize_size: tuple = (256, 256)
    
    # 动作平均参数
    use_action_ensembling: bool = False 

    # 日志相关参数
    local_log_dir: str = "/liujinxin/zhy/ICLR2026/eval/logs"
    save_videos: bool = False


class ActionEnsembler:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.prediction_history = deque(maxlen=horizon)
        self.current_step = 0
    
    def add_prediction(self, predicted_actions: np.ndarray):
        self.prediction_history.append((self.current_step, predicted_actions))
    
    def get_action(self) -> np.ndarray:
        if len(self.prediction_history) == 0:
            raise ValueError("No available predictions")
        
        predictions_for_current_step = []
        
        for pred_time, action_sequence in self.prediction_history:
            offset = self.current_step - pred_time
            
            if 0 <= offset < len(action_sequence):
                predictions_for_current_step.append(action_sequence[offset])
        
        if len(predictions_for_current_step) == 0:
            raise ValueError(f"current time step {self.current_step} has no available projections")
        
        averaged_action = np.mean(predictions_for_current_step, axis=0)
        
        return averaged_action
    
    def step(self):
        self.current_step += 1



def parse_args_and_create_config() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="LIBERO 评估脚本")
    
    parser.add_argument("--model_path", type=str, default="", 
                        help="模型路径")
    parser.add_argument("--action_norm_stats_path", type=str, default="", 
                        help="动作归一化统计数据路径")
    parser.add_argument("--task_suite_name", type=str, 
                        default=TaskSuite.LIBERO_SPATIAL,
                        choices=[suite.value for suite in TaskSuite],
                        help="任务套件名称")
    parser.add_argument("--num_steps_wait", type=int, default=10, 
                        help="等待步数")
    parser.add_argument("--num_trials_per_task", type=int, default=50, 
                        help="每个任务的试验次数")
    parser.add_argument("--initial_states_path", type=str, default="DEFAULT", 
                        help="初始状态路径")
    parser.add_argument("--env_img_res", type=int, default=256, 
                        help="环境图像分辨率")
    parser.add_argument("--local_log_dir", type=str, 
                        default="/liujinxin/zhy/ICLR2026/eval/logs", 
                        help="本地日志目录")
    parser.add_argument("--save_videos",action='store_true', 
                        help="保存视频")
    parser.add_argument("--use_action_ensembling", action='store_true',
                        help="是否使用动作平均策略")

    
    args = parser.parse_args()

    config = EvaluationConfig(
        model_path=args.model_path,
        action_norm_stats_path=args.action_norm_stats_path,
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials_per_task,
        initial_states_path=args.initial_states_path,
        env_img_res=args.env_img_res,
        local_log_dir=args.local_log_dir,
        save_videos=args.save_videos,
        use_action_ensembling=args.use_action_ensembling
    )
    
    return config


def eval_libero(cfg: EvaluationConfig) -> float:
    log_message(f"Loading model from {cfg.model_path}...", None)
    
    model = Qwen2_5_VL_3B_VLA(model_path=cfg.model_path, action_norm_stats_path=cfg.action_norm_stats_path)
    
    log_file, _ = setup_logging(cfg)
    
    # 记录动作平均策略的使用情况
    if cfg.use_action_ensembling:
        log_message(f"使用动作平均策略 (Temporal Ensembling) - horizon={NUM_ACTIONS_CHUNK}", log_file)
    else:
        log_message(f"使用传统动作队列策略 - open_loop_steps={cfg.num_open_loop_steps}", log_file)
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    needToEvaluate = [0,1,2,3,4,5,6,7,8,9]

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        if task_id not in needToEvaluate:
            continue
        total_episodes, total_successes = run_task(
            cfg,
            model,
            task_suite,
            task_id,
            total_episodes,
            total_successes,
            log_file
        )
    
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    

    log_message("final result:", log_file)
    log_message(f"All episodes: {total_episodes}", log_file)
    log_message(f"Total number of successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    
    
    if log_file:
        log_file.close()


def run_task(cfg: EvaluationConfig, model: Qwen2_5_VL_3B_VLA, task_suite, task_id,  total_episodes, total_successes, log_file):
    """Run evaluation for a single task"""
    task = task_suite.get_task(task_id)
    
    initial_states = task_suite.get_task_init_states(task_id)
    
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)
    
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        log_message(f"Start episode {task_episodes + 1}...", log_file)
        
        # 运行episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            initial_state,
            log_file
        )
        

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        

        if cfg.save_videos:
            save_rollout_video(
                replay_images, total_episodes, success=success, 
                task_description=task_description, log_file=log_file,
                output_dir=os.path.join(cfg.local_log_dir, "videos", cfg.model_path.split("/")[-1])
            )
        
        # 记录结果
        log_message(f"successes: {success}", log_file)
        log_message(f"Total number of episodes completed: {total_episodes}", log_file)
        log_message(f"Total number of successes so far: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    
    # 记录任务结果
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    
    log_message(f"Task success rate: {task_success_rate}", log_file)
    log_message(f"Total success rate so far: {total_success_rate}", log_file)
    
    
    return total_episodes, total_successes


def run_episode(cfg: EvaluationConfig, env, task_description, model, initial_state, log_file):
    """Run a single episode"""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK and not cfg.use_action_ensembling:
        log_message(f"Warning: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the constant NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK})！To achieve the best performance (speed and success rate), we recommend executing the complete action block.", log_file)
    
    if cfg.use_action_ensembling:
        action_ensembler = ActionEnsembler(horizon=NUM_ACTIONS_CHUNK)
    else:
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
    

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    
    # Run an episode
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action())
            t += 1
            continue
        
        observation, img = prepare_observation(obs, cfg.resize_size)
        replay_images.append(img)
        
        if cfg.use_action_ensembling:
            actions = model.generate_actions( task_description, observation["full_image"], observation["wrist_image"])
            action_ensembler.add_prediction(actions)
            action = action_ensembler.get_action()
            action_ensembler.step()
        else:
            if len(action_queue) == 0:
                actions = model.generate_actions(task_description, observation["full_image"], observation["wrist_image"])
                action_queue.extend(actions)
            
            action = action_queue.popleft()
        
        
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1
            
    
    return success, replay_images
    


def prepare_observation(obs, resize_size): 
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    def preprocess_libero_image(img_np, resize_size=(224, 224)):
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize(resize_size)
        aug_img_np = np.array(pil_img)
        return aug_img_np

    img_aug = preprocess_libero_image(img, resize_size)
    wrist_img_aug = preprocess_libero_image(wrist_img, resize_size)

    
    observation = {
        "full_image": ndarray_image_to_base64(img_aug),
        "wrist_image": ndarray_image_to_base64(wrist_img_aug)
    }
    
    return observation, img  


if __name__ == "__main__":
    config = parse_args_and_create_config()
    logging.info(f"Config: {config}")
    eval_libero(config)
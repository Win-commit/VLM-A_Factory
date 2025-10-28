import pathlib
from torch.utils.data import Dataset

LIBERO_RAW_DATA_PATH = pathlib.Path('/liujinxin/zhy/ICLR2026/datasets/libero/data/libero_all')
SYSTEM_PROMPT = "Analyze the input image and predict robot actions for the next {H} timesteps. Each action has Ddimensions. Output a single sequence of {H}*{D} integers (0-{B} each), representing the {H} timesteps sequentially. Provide only space-separated numbers. Nothing else."


eposides = LIBERO_RAW_DATA_PATH.glob('*/')

for episode in eposides:
    main_image_folder = episode / 'images'
    main_image_files = list(main_image_folder.glob('*.jpg'))
    gripper_image_folder = episode / 'gripper_images'
    gripper_image_files = list(gripper_image_folder.glob('*.jpg'))
    action_folder = episode / 'actions'
    action_files = list(action_folder.glob('*.npy'))
    task_instruction_file = episode / 'instruction.txt'
    task_instruction = task_instruction_file.read_text()





class LiberoVLA0(Dataset):
    def __init__(self, eposide_path: pathlib.Path):
        pass





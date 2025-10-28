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



'''
TODO: 把libero数据集处理成经典llm微调的格式,任意时刻t0应该对应于字符串输入:
message: [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": main_image_files
            },
            {
                "type": "image",
                "image": gripper_image_files
            },
            {
                "type": "text",
                "text": task_instruction
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": action_files(H*D个范围在[0,B]的整数，归一化得到)
            }
        ]
    }
]
'''

class LiberoVLA0(Dataset):
    def __init__(self, eposide_path: pathlib.Path):
        pass





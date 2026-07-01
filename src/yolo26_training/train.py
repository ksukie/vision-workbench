"""最基础的 YOLO26 训练脚本。"""

import os
import sys


# 1. 自动找到项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# 2. 让 Python 可以找到本项目代码和 YOLO26 源码
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "yolo26_source"))

from yolo26_training.infrastructure import YoloDetectionDatasetValidator
from yolo26_training.configuration import Yolo26TrainingConfig
from vision_workbench.troubleshooting import DATASETS_AND_TRAINING, DEEP_LEARNING_DEPENDENCIES, MODELS_AND_WEIGHTS, with_help


# 3. 修改这里：你的数据集 data.yaml 路径
DATA_YAML = r"C:\path\to\your_dataset\data.yaml"

# 4. 修改这里：训练任务，可选 detect / segment / semantic
TASK = "detect"

# 5. 修改这里：选择一个基础模型
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo26_models", "yolo26n.pt")

# 6. 修改这里：训练参数
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = "auto"  # 可选：auto / cpu / cuda / mps / 0
WORKERS = 8

# 7. 训练结果保存位置
PROJECT_DIR = os.path.join(PROJECT_ROOT, "runs", "yolo26_training")
RUN_NAME = "my_yolo26_train"


def main():
    if DATA_YAML == r"C:\path\to\your_dataset\data.yaml":
        print(with_help("请先打开 train.py，修改 DATA_YAML 为你自己的 data.yaml 路径。", DATASETS_AND_TRAINING))
        return

    print("开始检查数据集...")
    validator = YoloDetectionDatasetValidator(Yolo26TrainingConfig())
    report = validator.validate(DATA_YAML, task=TASK)
    print(report.to_text())

    if not report.ok:
        print(with_help("数据集不符合要求，训练已停止。", DATASETS_AND_TRAINING))
        return

    if not os.path.exists(MODEL_PATH):
        print(with_help(f"模型文件不存在：\n{MODEL_PATH}", MODELS_AND_WEIGHTS))
        return

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(with_help(f"无法导入 YOLO26 运行环境。\n{exc}\n请在项目根目录执行：pip install -r requirements-yolo26.txt", DEEP_LEARNING_DEPENDENCIES))
        return

    print("开始训练...")
    model = YOLO(MODEL_PATH)

    train_device = None
    if DEVICE != "auto":
        train_device = DEVICE

    try:
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=train_device,
            workers=WORKERS,
            project=PROJECT_DIR,
            name=RUN_NAME,
        )
    except Exception as exc:
        print(with_help(f"训练失败：{exc}", DATASETS_AND_TRAINING))
        return

    print("训练完成。")
    print("结果目录：")
    print(os.path.join(PROJECT_DIR, RUN_NAME))


if __name__ == "__main__":
    main()

from mmengine.config import Config
from mmengine.runner import Runner


def main() -> None:
    """Entry point for training an object detection model."""
    config_path = "config/object_detection.py"

    cfg = Config.fromfile(config_path)
    runner = Runner.from_cfg(cfg)

    runner.train()


if __name__ == "__main__":
    main()

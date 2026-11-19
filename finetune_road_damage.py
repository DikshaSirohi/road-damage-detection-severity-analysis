
from pathlib import Path
from ultralytics import YOLO
import torch
import multiprocessing
import argparse

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = PROJECT_DIR / "road_damage_yolo" / "data.yaml"
RUNS_DIR = PROJECT_DIR / "runs"

V1_WEIGHTS = (
    RUNS_DIR / "road_damage_yolo26n_v1" / "weights" / "best.pt"
)

RUN_NAME = "road_damage_yolo26n_v2"
RUN_DIR = RUNS_DIR / RUN_NAME
LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"
BEST_CHECKPOINT = RUN_DIR / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("\n=== GPU INFORMATION ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    try:
        if args.resume:
            if not LAST_CHECKPOINT.is_file():
                raise FileNotFoundError(LAST_CHECKPOINT)

            print("\n=== RESUMING V2 TRAINING ===")
            print(f"Checkpoint: {LAST_CHECKPOINT}")

            model = YOLO(str(LAST_CHECKPOINT))
            model.train(resume=True)

        else:
            if not V1_WEIGHTS.is_file():
                raise FileNotFoundError(V1_WEIGHTS)

            if not DATA_YAML.is_file():
                raise FileNotFoundError(DATA_YAML)

            if RUN_DIR.exists():
                raise RuntimeError(
                    "V2 training folder already exists. "
                    "Use --resume if training was interrupted."
                )

            print("\n=== V2 FINE-TUNING CONFIGURATION ===")
            print(f"Starting weights: {V1_WEIGHTS}")
            print("Additional epochs: 20")
            print("Optimizer: AdamW")
            print("Learning rate: 0.0003")
            print("Batch size: 4")
            print("Image size: 640")
            print("Early stopping patience: 5")
            print(f"Output folder: {RUN_DIR}")

            model = YOLO(str(V1_WEIGHTS))

            model.train(
                data=str(DATA_YAML),
                epochs=20,
                imgsz=640,
                batch=4,
                device=0,
                workers=2,

                pretrained=True,
                resume=False,

                optimizer="AdamW",
                lr0=0.0003,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=1.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.0,
                cos_lr=False,

                amp=False,
                cache=False,
                patience=5,

                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                close_mosaic=5,

                seed=42,
                deterministic=True,

                val=True,
                plots=True,
                save=True,
                save_period=5,

                project=str(RUNS_DIR),
                name=RUN_NAME,
                exist_ok=False,
                verbose=True
            )

        print("\n=== V2 TRAINING FINISHED ===")
        print(f"Best model: {BEST_CHECKPOINT}")
        print(f"Last model: {LAST_CHECKPOINT}")
        print(f"Results: {RUN_DIR / 'results.csv'}")

    except KeyboardInterrupt:
        print("\n=== V2 TRAINING INTERRUPTED ===")

        if LAST_CHECKPOINT.is_file():
            print(f"Checkpoint: {LAST_CHECKPOINT}")
            print("To resume:")
            print("python finetune_road_damage.py --resume")
        else:
            print("No completed checkpoint saved yet.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

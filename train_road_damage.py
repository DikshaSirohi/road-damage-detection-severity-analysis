
from pathlib import Path
from ultralytics import YOLO
import torch
import multiprocessing
import argparse

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = PROJECT_DIR / "road_damage_yolo" / "data.yaml"
MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
RUNS_DIR = PROJECT_DIR / "runs"

RUN_NAME = "road_damage_yolo26n_v1"
RUN_DIR = RUNS_DIR / RUN_NAME

LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"
BEST_CHECKPOINT = RUN_DIR / "weights" / "best.pt"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size"
    )

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("\n=== GPU INFORMATION ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    try:

        # Resume an interrupted training run
        if args.resume:

            if not LAST_CHECKPOINT.is_file():
                raise FileNotFoundError(
                    f"No checkpoint found: {LAST_CHECKPOINT}"
                )

            print("\n=== RESUMING TRAINING ===")
            print(f"Checkpoint: {LAST_CHECKPOINT}")

            model = YOLO(str(LAST_CHECKPOINT))

            model.train(resume=True)

        # Start fresh training
        else:

            if not DATA_YAML.is_file():
                raise FileNotFoundError(DATA_YAML)

            if not MODEL_PATH.is_file():
                raise FileNotFoundError(MODEL_PATH)

            if RUN_DIR.exists():
                raise RuntimeError(
                    "A training run already exists.\n"
                    "Use --resume to continue an interrupted run."
                )

            print("\n=== TRAINING CONFIGURATION ===")
            print(f"Model: {MODEL_PATH}")
            print(f"Dataset: {DATA_YAML}")
            print(f"Maximum epochs: {args.epochs}")
            print(f"Batch size: {args.batch}")
            print("Image size: 640")
            print("Early stopping patience: 5")
            print(f"Output folder: {RUN_DIR}")

            model = YOLO(str(MODEL_PATH))

            model.train(
                data=str(DATA_YAML),

                epochs=args.epochs,
                imgsz=640,
                batch=args.batch,

                device=0,
                workers=2,

                pretrained=True,
                optimizer="auto",

                amp=True,
                cache=False,

                patience=5,

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

        print("\n=== TRAINING FINISHED ===")
        print(f"Best model: {BEST_CHECKPOINT}")
        print(f"Last model: {LAST_CHECKPOINT}")
        print(f"Results: {RUN_DIR / 'results.csv'}")

    except KeyboardInterrupt:

        print("\n\n=== TRAINING INTERRUPTED ===")

        if LAST_CHECKPOINT.is_file():

            print("Previous completed epoch is saved.")
            print(f"Checkpoint: {LAST_CHECKPOINT}")

            print("\nTo resume training, run:")
            print("python train_road_damage.py --resume")

        else:

            print("No completed epoch checkpoint found.")
            print("Training must be restarted.")

        if BEST_CHECKPOINT.is_file():
            print(f"\nBest model available: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

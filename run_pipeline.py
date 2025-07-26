import subprocess
import sys

# Helper to run a python script with arguments
def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

train_epochs = 40
attack_epochs = 30

if __name__ == "__main__":
    # Train for both datasets (baseline and watermark)
    for dataset in ["mnist", "fashionmnist"]:
        # Baseline
        run_script("src/train.py", ["--dataset", dataset, "--epochs", str(train_epochs), "--batch_size", "100", "--learning_rate", "0.001", "--early_stopping", "--patience", "5"])
        # Baseline from scratch
        run_script("src/train.py", ["--dataset", dataset, "--trainFromScratch", "--epochs", str(train_epochs), "--batch_size", "100", "--learning_rate", "0.001", "--early_stopping", "--patience", "5", "--model_path", "scratch_baseline.pth"])
        # Watermark
        run_script("src/train.py", ["--dataset", dataset, "--train_watermark", "--epochs", str(train_epochs), "--batch_size", "100", "--learning_rate", "0.001", "--early_stopping", "--patience", "5"])
        # Watermark from scratch
        run_script("src/train.py", ["--dataset", dataset, "--train_watermark", "--trainFromScratch", "--epochs", str(train_epochs), "--batch_size", "100", "--learning_rate", "0.001", "--early_stopping", "--patience", "5", "--model_path", "scratch_watermarked.pth"])
    # Attack for both datasets
    for dataset in ["mnist", "fashionmnist"]:
        for attack in ["ftll", "ftal", "rtll", "rtal"]:
            # Attacks on normal watermarked
            run_script("src/attack.py", ["--dataset", dataset, "--attack", attack, "--num_epochs", str(attack_epochs), "--lr", "0.001"])
            # Attacks on scratch watermarked
            run_script("src/attack.py", ["--dataset", dataset, "--attack", attack, "--trainFromScratch", "--num_epochs", str(attack_epochs), "--lr", "0.001"])
    # Evaluate for all
    model_types = [
        "baseline",
        "scratch_baseline",
        "watermarked",
        "scratch_watermarked",
        "ftll_attacked",
        "scratch_ftll_attacked",
        "ftal_attacked",
        "scratch_ftal_attacked",
        "rtll_attacked",
        "scratch_rtll_attacked",
        "rtal_attacked",
        "scratch_rtal_attacked"
    ]
    for dataset in ["mnist", "fashionmnist"]:
        for model_type in model_types:
            run_script("src/evaluate.py", ["--dataset", dataset, "--model_type", model_type])
    print("Pipeline finished.")

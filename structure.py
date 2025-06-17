
import os

PROJECT_NAME = "ml_watermarking_project"

# Define the directory structure
# Using relative paths from the project root
dirs_to_create = [
    "config",
    "data/raw",
    "data/processed",
    "notebooks",
    "report/figures",
    "results/example_experiment_whitebox_googlenet_cifar10/logs",
    "results/example_experiment_whitebox_googlenet_cifar10/models",
    "results/example_experiment_whitebox_googlenet_cifar10/plots",
    "results/example_experiment_blackbox_squeezenet_mnist/logs",      # Example for black-box scenario
    "results/example_experiment_blackbox_squeezenet_mnist/models",
    "results/example_experiment_blackbox_squeezenet_mnist/plots",
    "src/models",
    "src/watermarking/your_chosen_technique_name_here", # IMPORTANT: Rename this!
    "src/evaluation",
    "src/utils",
    "tests",
]

# Define empty files to create (often for Python packages or placeholders)
# Using tuples: (path_relative_to_project_root, optional_content_or_comment)
files_to_create = [
    (".gitignore", "*.pyc\n__pycache__/\n.DS_Store\n/data/raw/*\n/data/processed/*\n/results/*/\n!/results/.gitkeep\n*.pth\n*.pt\n*.onnx\n/report/*.pdf\n*.log"),
    ("README.md", f"# {PROJECT_NAME}\n\nProject for Topic 2.2.3: Watermarking ML/DL models."),
    ("requirements.txt", "# Add your Python dependencies here, e.g.:\n# torch\n# torchvision\n# numpy\n# matplotlib\n# scikit-learn\n# pyyaml"),

    ("config/experiment_params.yaml", "# General experiment settings\n# Example:\n# dataset: CIFAR-10\n# architecture: GoogleNet\n# learning_rate: 0.001\n# epochs: 50\n# batch_size: 64\n# attack_type: fine_tuning\n# attack_params:\n#   epochs: 10\n#   learning_rate: 0.0001"),
    ("config/watermark_params.yaml", "# Parameters specific to the chosen watermarking technique\n# Example for Uchida et al.:\n# watermark_bit_length: 64\n# embedding_strength: 0.1\n# \n# Example for Backdooring (Adi et al.):\n# trigger_pattern_path: \"data/triggers/my_trigger.png\"\n# target_label: 7\n# num_trigger_samples: 100"),

    ("results/.gitkeep", "# Keep the results directory, even if empty initially"),
    ("results/example_experiment_whitebox_googlenet_cifar10/metrics.json", "{\n  \"baseline_accuracy\": null,\n  \"watermarked_accuracy\": null,\n  \"watermark_extracted_before_attack\": null,\n  \"accuracy_after_attack\": null,\n  \"watermark_extracted_after_attack\": null\n}"),
    ("results/example_experiment_blackbox_squeezenet_mnist/metrics.json", "{\n  \"baseline_accuracy\": null,\n  \"watermarked_accuracy\": null,\n  \"watermark_extracted_before_attack\": null,\n  \"accuracy_after_attack\": null,\n  \"watermark_extracted_after_attack\": null\n}"),


    ("src/__init__.py", ""),
    ("src/data_loader.py", "#!/usr/bin/env python3\n# Functions for loading and preprocessing datasets (MNIST, FashionMNIST, CIFAR-10, etc.)\n"),
    ("src/models/__init__.py", ""),
    ("src/models/base_model.py", "#!/usr/bin/env python3\n# (Optional) Base class for models\n"),
    ("src/models/googlenet.py", "#!/usr/bin/env python3\n# GoogleNet implementation or wrapper\n"),
    ("src/models/squeezenet.py", "#!/usr/bin/env python3\n# SqueezeNet implementation or wrapper\n"),
    ("src/models/densenet.py", "#!/usr/bin/env python3\n# DenseNet implementation or wrapper\n"),

    ("src/watermarking/__init__.py", ""),
    ("src/watermarking/your_chosen_technique_name_here/__init__.py", ""), # IMPORTANT: Matches dir name
    ("src/watermarking/your_chosen_technique_name_here/embedder.py", "#!/usr/bin/env python3\n# Logic for embedding the watermark using [Your Chosen Technique]\n"),
    ("src/watermarking/your_chosen_technique_name_here/extractor.py", "#!/usr/bin/env python3\n# Logic for extracting/verifying the watermark for [Your Chosen Technique]\n"),
    ("src/watermarking/your_chosen_technique_name_here/utils.py", "#!/usr/bin/env python3\n# Helper functions specific to [Your Chosen Technique]\n"),

    ("src/evaluation/__init__.py", ""),
    ("src/evaluation/metrics.py", "#!/usr/bin/env python3\n# Functions to calculate accuracy, watermark success rate, etc.\n"),
    ("src/evaluation/attacks.py", "#!/usr/bin/env python3\n# Implementations of attacks (e.g., fine-tuning, pruning, transfer learning)\n"),
    ("src/evaluation/evaluators.py", "#!/usr/bin/env python3\n# Scripts/classes to run fidelity, effectiveness, robustness checks\n"),

    ("src/utils/__init__.py", ""),
    ("src/utils/general_utils.py", "#!/usr/bin/env python3\n# General helper functions (saving/loading models, logging, etc.)\n"),
    ("src/utils/plot_utils.py", "#!/usr/bin/env python3\n# Helper functions for plotting results\n"),

    ("src/train_baseline.py", "#!/usr/bin/env python3\n# Script to train a model without watermarking\nif __name__ == '__main__':\n    print(\"Training baseline model...\")\n"),
    ("src/embed_watermark.py", "#!/usr/bin/env python3\n# Script to embed watermark into a trained model\nif __name__ == '__main__':\n    print(\"Embedding watermark...\")\n"),
    ("src/verify_watermark.py", "#!/usr/bin/env python3\n# Script to verify watermark from a model\nif __name__ == '__main__':\n    print(\"Verifying watermark...\")\n"),
    ("src/run_attack.py", "#!/usr/bin/env python3\n# Script to perform an attack on a watermarked model\nif __name__ == '__main__':\n    print(\"Running attack...\")\n"),
    ("src/run_experiment.py", "#!/usr/bin/env python3\n# Main script to orchestrate an entire experiment (train, watermark, evaluate)\nif __name__ == '__main__':\n    print(\"Running full experiment...\")\n"),

    # Minimal valid ipynb structure for notebooks
    ("notebooks/01_data_exploration.ipynb", "{\n \"cells\": [],\n \"metadata\": {},\n \"nbformat\": 4,\n \"nbformat_minor\": 5\n}"),
    ("notebooks/02_model_training_baseline.ipynb", "{\n \"cells\": [],\n \"metadata\": {},\n \"nbformat\": 4,\n \"nbformat_minor\": 5\n}"),
    ("notebooks/03_watermark_embedding_dev.ipynb", "{\n \"cells\": [],\n \"metadata\": {},\n \"nbformat\": 4,\n \"nbformat_minor\": 5\n}"),
    ("notebooks/04_evaluation_dev.ipynb", "{\n \"cells\": [],\n \"metadata\": {},\n \"nbformat\": 4,\n \"nbformat_minor\": 5\n}"),
    ("notebooks/05_results_visualization.ipynb", "{\n \"cells\": [],\n \"metadata\": {},\n \"nbformat\": 4,\n \"nbformat_minor\": 5\n}"),

    ("tests/__init__.py", ""),
    ("tests/test_data_loader.py", "#!/usr/bin/env python3\n# Unit tests for data_loader.py\nimport unittest\n\nclass TestDataLoader(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n"),
    ("tests/test_models.py", "#!/usr/bin/env python3\n# Unit tests for models\nimport unittest\n\nclass TestModels(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n"),
    ("tests/test_watermarking_chosen_technique.py", "#!/usr/bin/env python3\n# Unit tests for your chosen watermarking technique\n# Remember to rename this file if you rename the technique folder!\nimport unittest\n\nclass TestWatermarking(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n"),
]

def create_structure():
    # Create the root project directory
    try:
        os.makedirs(PROJECT_NAME, exist_ok=True)
        print(f"Created project root: {PROJECT_NAME}")
    except OSError as e:
        print(f"Error creating project root {PROJECT_NAME}: {e}")
        return

    # Create subdirectories
    for dir_path_rel in dirs_to_create:
        full_path = os.path.join(PROJECT_NAME, dir_path_rel)
        try:
            os.makedirs(full_path, exist_ok=True)
            print(f"  Created directory: {full_path}")
        except OSError as e:
            print(f"  Error creating directory {full_path}: {e}")

    # Create empty/template files
    for file_path_rel, content in files_to_create:
        full_path = os.path.join(PROJECT_NAME, file_path_rel)
        try:
            # Ensure parent directory exists for the file (it should, from the step above, but good practice)
            parent_dir = os.path.dirname(full_path)
            if parent_dir: # Check if it's not a top-level file in PROJECT_NAME
                os.makedirs(parent_dir, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                if content:
                    f.write(content) # Add newline for cleaner files if not already in content
                    if not content.endswith('\n'):
                        f.write('\n')

            print(f"  Created file:      {full_path}")
            # Make .py files executable (on Unix-like systems)
            if full_path.endswith(".py"):
                try:
                    os.chmod(full_path, 0o755)
                except OSError:
                    pass # Silently pass on systems where chmod might not be applicable or fail (e.g. Windows)
        except OSError as e:
            print(f"  Error creating file {full_path}: {e}")

    print("\nProject structure created successfully!")
    print(f"IMPORTANT: Remember to rename 'src/watermarking/your_chosen_technique_name_here'")
    print(f"  and potentially 'tests/test_watermarking_chosen_technique.py' to match your chosen method.")
    print(f"You might also want to customize 'results/' with actual experiment names instead of the examples.")

if __name__ == "__main__":
    create_structure()


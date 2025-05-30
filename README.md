<<<<<<< HEAD
# ChessViT: A Vision Transformer for Chess

## Overview

ChessViT is a project that implements a Vision Transformer (ViT) model for playing and analyzing chess. It includes a comprehensive suite of tools for data preprocessing, model training, and evaluation. The model leverages memory-mapped datasets for efficient data loading and supports mixed-precision training.

## Features

*   **Vision Transformer for Chess:** Implements a ViT model tailored for chess, processing the board state as an image.
*   **Efficient Data Loading:** Utilizes memory-mapped chunk datasets (`ChunkMmapDataset`) for high-throughput, low-overhead data loading, compatible with `DataLoader(pin_memory=True)`.
*   **Data Preprocessing Pipeline:**
    *   `pgn_splitter.py`: Fast, parallel PGN sharder that splits large PGN collections (including compressed archives like `.tar.gz`, `.pgn.gz`) into smaller, manageable PGN chunks. Includes filtering for Lichess games based on Elo, termination, and game type.
    *   `preprocess_tensors.py`: Converts PGN chunks into fixed-size `.npz` tensor chunks suitable for training. Features deterministic multi-pass mix-shuffling and structured logging.
*   **Flexible Training:**
    *   `train.py`: Training script with support for mixed-precision (AMP) training, gradient clipping, and rolling-average metrics.
    *   Configurable loss functions and weights.
    *   TensorBoard and Matplotlib logging for monitoring training progress.
    *   Checkpointing with `safetensors` for model weights and PyTorch files for optimizer/scheduler state.
*   **Model Configuration:** Uses YAML files for easy configuration of model architecture, optimizer, dataset, and runtime parameters. Multiple configurations (tiny, small, medium, large) are provided.
*   **Utilities:**
    *   `dist_init.py`: Generates an initial 65x65 distance bias matrix for the ViT's attention mechanism.
    *   `mapping.py`: Defines move-to-plane and plane-to-move mappings for the policy head.
    *   `losses.py`: Contains definitions for various loss functions used during training (WDL, material, ply, policy, contrastive).
    *   `create_test_split_and_analyze.py`: Script to create a test split from the dataset and analyze WDL and policy distributions.

## Directory Structure

```
ChessViT/
├── configs/                  # YAML configuration files for different model sizes
│   ├── large.yaml
│   ├── medium.yaml
│   ├── small.yaml
│   └── tiny.yaml
├── chess_dataset.py          # Memory-mapped dataset loader
├── chess_vit.py              # ViT model definition
├── create_test_split_and_analyze.py # Script for test set creation and analysis
├── dist_init.py              # Generates distance matrix for attention bias
├── losses.py                 # Loss function definitions
├── mapping.py                # Move to policy plane mapping
├── pgn_splitter.py           # Splits PGN files into chunks
├── preprocess_tensors.py     # Converts PGN chunks to tensor chunks (.npz)
├── stream_reader.py          # Reads games from PGN files and archives
├── stream_worker.py          # Encodes PGN game data into features
├── test_preprocessing.py     # Test script for the preprocessing pipeline
├── train.py                  # Main training script
└── README.md                 # This file
```

## Setup / Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ChessViT
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```
3.  **Install dependencies:**
    While a `requirements.txt` is not explicitly provided in the context, typical dependencies would include:
    ```bash
    pip install torch torchvision torchaudio numpy pyyaml safetensors python-chess matplotlib
    ```
    Ensure you install the PyTorch version compatible with your CUDA setup if using a GPU.

## Data Preprocessing

The data preprocessing pipeline involves two main steps:

1.  **Splitting PGNs into Chunks (`pgn_splitter.py`):**
    This script takes raw PGN files (or archives like `.tar.gz`) and splits them into smaller PGN files, each containing a fixed number of games. This is useful for managing large datasets.
    ```bash
    python pgn_splitter.py \
           --in-dir path/to/raw_pgns \
           --out-dir path/to/pgn_shards \
           --games-per-chunk 2500 \
           --workers <num_workers>
    ```
    The `raw_pgns` directory should have subdirectories like `lichess` and `lc0` if you want to process games from different sources separately.

2.  **Converting PGN Chunks to Tensor Chunks (`preprocess_tensors.py`):**
    This script takes the PGN shards generated by `pgn_splitter.py` (or other PGN files) and converts them into `.npz` files containing tensors. These `.npz` files are then used for training.
    ```bash
    python preprocess_tensors.py \
           --raw-dir path/to/pgn_shards \
           --out-dir path/to/tensor_chunks \
           --chunk-size 16384 \
           --jobs <num_workers> \
           --mix-passes 3 \
           --worker-buffer-multiplier 4
    ```
    The `pgn_shards` directory structure should also contain `lichess` and `lc0` subdirectories if your data is organized that way. The script will place the output tensor chunks into corresponding subdirectories in `path/to/tensor_chunks`.

    *   `--worker-buffer-multiplier`: Controls how many `chunk-size` blocks are buffered in RAM by each worker before an internal shuffle and flush to disk. Higher values use more RAM per worker but can improve shuffling and reduce I/O.
    *   `--skip-final-shuffle-pass`: If the `worker-buffer-multiplier` is high enough that chunks written by workers are already well-shuffled, this flag can be used to skip the final disk-based merge-shuffle passes, saving time.

## Training

Once the data is preprocessed into tensor chunks:

1.  **Prepare the Distance Matrix:**
    The `dist_init.py` script generates `dist_init.npy`, which is used as an attention bias in the ViT model. Run this once:
    ```bash
    python dist_init.py
    ```
    This will create `dist_init.npy` in the current directory. Ensure the `distance_matrix_path` in your chosen config YAML file points to this file.

2.  **Run the Training Script:**
    Use `train.py` with a configuration file:
    ```bash
    python train.py --config configs/medium.yaml
    ```
    Adjust parameters within the YAML file (e.g., `raw_data_dir` in the `dataset` section should point to your `tensor_chunks` directory) or override them via command-line arguments if the script supports it.

    *   The training script handles loading the model, optimizer, scheduler, and data.
    *   It logs metrics to TensorBoard and can save Matplotlib plots of loss curves.
    *   Checkpoints are saved periodically.

## Model Configuration

Model architecture, training parameters, dataset paths, and logging options are defined in YAML files within the `configs/` directory. Key sections in the config files include:

*   `model`: Defines ViT architecture (dimensions, depth, heads), paths to auxiliary files like the distance matrix, and parameters for features like CLS token pooling and dropout.
*   `loss_weights`: Specifies weights for different components of the loss function (policy, value, moves left, etc.).
*   `optimiser`: Configures the optimizer (e.g., AdamW) and learning rate scheduler.
*   `dataset`: Specifies the path to the training data (tensor chunks), batch size, number of workers, and other data-related settings. The `type` can be `tensor` (for `ChunkMmapDataset`).
*   `runtime`: Defines runtime settings like device (CPU/CUDA), precision (e.g., fp16 for mixed-precision), gradient accumulation, and checkpointing frequency.
*   `logging`: Configures output directories for runs, TensorBoard, and Matplotlib logs.

## Utilities

*   **`create_test_split_and_analyze.py`**: This script helps in creating a hold-out test set by moving a specified number of `.npz` files from your training data directories (e.g., `data/lichess`, `data/lc0`) to a separate test directory (e.g., `test/lichess`, `test/lc0`). It then analyzes and compares the WDL (Win-Draw-Loss) and policy plane distributions between the newly created test set and a Simple Random Sample (SRS) from the remaining training data. This is useful for checking if the test set is representative.
    ```bash
    python create_test_split_and_analyze.py \
           --base-data-dir path/to/tensor_chunks \
           --base-test-dir path/to/test_set_output \
           --output-plot-dir path/to/analysis_plots \
           --num-files-to-move 3 \
           --num-train-srs-positions 500 \
           --srs-workers 4
    ```

## Testing

A simple test for the preprocessing pipeline can be run using:
```bash
python test_preprocessing.py
```
This script creates dummy PGN files, runs `preprocess_tensors.py` on them with minimal settings, and checks if output files are generated as expected.

---

This README provides a general guide. You might need to adjust paths, parameters, and commands based on your specific setup and dataset. 
=======
# ChessViT
>>>>>>> origin/main

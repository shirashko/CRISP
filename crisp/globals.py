import os
import torch
import numpy as np
import random

# Paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA_SAE_CACHE_PATH = os.path.join(PROJECT_PATH, "crisp", "gemma_sae_cache")
LLAMA_3_1_SAE_CACHE_PATH = os.path.join(PROJECT_PATH, "crisp", "llama_sae_cache")
QWEN35_SAE_CACHE_PATH = os.path.join(PROJECT_PATH, "crisp", "qwen_sae_cache")
SAE_LLAMA_3_1_8B = "fnlp/Llama3_1-8B-Base-LXR-8x"
SAE_GEMMA_2_2B = "google/gemma-2-2b"
SAE_QWEN35_2B = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
GEMMA_2_2B = "google/gemma-2-2b"
QWEN35_2B = "Qwen/Qwen3.5-2B"
QWEN35_2B_BASE = "Qwen/Qwen3.5-2B-Base"
LLAMA_3_1_8B_IT = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA_2_2B_IT = "google/gemma-2-2b-it"
QWEN35_2B_IT = QWEN35_2B
SAE_GEMMA_2_2B_IT = SAE_GEMMA_2_2B      # instruct model uses same SAE weights as base
SAE_LLAMA_3_1_8B_IT = SAE_LLAMA_3_1_8B  # instruct model uses same SAE weights as base
SAE_QWEN35_2B_IT = SAE_QWEN35_2B        # instruct model uses same SAE weights as base

SEED = 0

def set_seed(seed=SEED, deterministic_cudnn=True):
    """
    Set seeds for reproducibility.

    Args:
        seed: Integer seed for random number generators
        deterministic_cudnn: Whether to make cuDNN deterministic (may reduce performance)
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch operations deterministic
    if deterministic_cudnn:
        # For complete determinism (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ settings for determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            # Available from PyTorch 1.8
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # Older PyTorch versions
            torch.set_deterministic(True)

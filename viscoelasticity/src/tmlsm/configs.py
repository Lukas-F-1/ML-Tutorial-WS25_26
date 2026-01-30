"""Experiment configurations for viscoelasticity models."""

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# Material Parameters (fixed for all experiments)
# =============================================================================
MATERIAL_PARAMS = {
    "E_infty": 0.5,
    "E": 2.0,
    "eta": 1.0,
}


# =============================================================================
# Model Types
# =============================================================================
ModelType = Literal["simple_rnn", "maxwell", "maxwell_nn", "gsm"]

ALL_MODELS: list[ModelType] = ["simple_rnn", "maxwell", "maxwell_nn", "gsm"]
TRAINABLE_MODELS: list[ModelType] = ["simple_rnn", "maxwell_nn", "gsm"]


# =============================================================================
# Experiment Configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str

    # Load cases: list of (A, omega) tuples
    train_loadcases: list[tuple[float, float]]
    test_loadcases: list[tuple[float, float]]

    # Time discretization
    n_timesteps: int = 100  # points per period
    n_periods: int = 1  # number of periods for training

    # Training parameters
    train_steps: int = 100_000
    log_every: int = 100

    # Models to run (default: all trainable)
    models: list[ModelType] = field(default_factory=lambda: TRAINABLE_MODELS.copy())

    # Include relaxation test
    test_relaxation: bool = True


# =============================================================================
# Predefined Experiments
# =============================================================================

# Baseline: Train on (1,1), test on all
BASELINE = ExperimentConfig(
    name="baseline",
    description="Train on (A=1, omega=1), test on all load cases",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Multi-loadcase training
MULTI_LOADCASE = ExperimentConfig(
    name="multi_loadcase",
    description="Train on multiple load cases for better generalization",
    train_loadcases=[(1.0, 1.0), (1.0, 2.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# All loadcases for training
ALL_LOADCASES = ExperimentConfig(
    name="all_loadcases",
    description="Train on all load cases",
    train_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Fine timesteps (higher resolution)
FINE_TIMESTEPS = ExperimentConfig(
    name="fine_timesteps",
    description="Higher time resolution (n=200)",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_timesteps=200,
)

# Coarse timesteps (lower resolution)
COARSE_TIMESTEPS = ExperimentConfig(
    name="coarse_timesteps",
    description="Lower time resolution (n=50)",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_timesteps=50,
)

# Multiple periods for training
MULTI_PERIOD = ExperimentConfig(
    name="multi_period",
    description="Train on 3 periods instead of 1",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_periods=3,
)

# Different training loadcase
TRAIN_ON_OMEGA2 = ExperimentConfig(
    name="train_omega2",
    description="Train on (A=1, omega=2) instead of (1,1)",
    train_loadcases=[(1.0, 2.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Higher amplitude training
TRAIN_HIGH_AMPLITUDE = ExperimentConfig(
    name="train_high_amplitude",
    description="Train on (A=2, omega=3)",
    train_loadcases=[(2.0, 3.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)


# =============================================================================
# Experiment Collections
# =============================================================================

# Core experiments (answer the main questions from the task)
CORE_EXPERIMENTS = [
    BASELINE,
    MULTI_LOADCASE,
    ALL_LOADCASES,
]

# Timestep experiments
TIMESTEP_EXPERIMENTS = [
    COARSE_TIMESTEPS,
    BASELINE,  # n=100 as reference
    FINE_TIMESTEPS,
]

# Training loadcase experiments
LOADCASE_EXPERIMENTS = [
    BASELINE,
    TRAIN_ON_OMEGA2,
    TRAIN_HIGH_AMPLITUDE,
]

# All experiments
ALL_EXPERIMENTS = [
    BASELINE,
    MULTI_LOADCASE,
    ALL_LOADCASES,
    FINE_TIMESTEPS,
    COARSE_TIMESTEPS,
    MULTI_PERIOD,
    TRAIN_ON_OMEGA2,
    TRAIN_HIGH_AMPLITUDE,
]

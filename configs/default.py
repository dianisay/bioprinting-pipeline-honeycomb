"""Default configuration for the bioprinting pipeline."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    fuseg_dir: str = "data/fuseg"
    synthetic_dir: str = "data/synthetic"
    num_synthetic: int = 2000
    num_boundary_points: int = 64  # N radii in polar representation
    image_size: int = 256
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class EncoderConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    feature_dim: int = 2048
    projection_dim: int = 256
    transformer_layers: int = 6
    transformer_heads: int = 8
    dropout: float = 0.1


@dataclass
class PolarDecoderConfig:
    num_radii: int = 64
    loss_weight_centroid: float = 1.0
    loss_weight_radii: float = 1.0
    loss_weight_points: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class RobotConfig:
    dof: int = 8  # 2 prismatic (gantry) + 6 revolute (UR5)
    gantry_range_x: float = 1.0  # meters
    gantry_range_y: float = 1.0
    pid_kp: float = 2.0
    pid_ki: float = 0.5
    pid_kd: float = 0.1
    convergence_threshold: float = 1e-3  # radians
    control_rate: float = 50.0  # Hz


@dataclass
class TrajectoryConfig:
    hex_cell_size: float = 3.0  # mm
    nozzle_rise_height: float = 20.0  # mm
    milp_timeout: int = 60  # seconds
    num_layers: int = 3
    layer_height: float = 0.3  # mm


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    polar_decoder: PolarDecoderConfig = field(default_factory=PolarDecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

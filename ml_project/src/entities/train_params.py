from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="SupportVectorMachine")
    random_state: int = field(default=123)

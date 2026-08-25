from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path = Path("data/generated")
    artifact_dir: Path = Path("artifacts")

    @property
    def transactions(self) -> Path:
        return self.data_dir / "transactions.csv"

    @property
    def model(self) -> Path:
        return self.artifact_dir / "isolation_forest.joblib"

    @property
    def scores(self) -> Path:
        return self.artifact_dir / "scored_transactions.csv"


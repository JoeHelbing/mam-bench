"""Simulation-owned records and atomic result files."""

import json
from pathlib import Path

from pydantic import BaseModel


class ArtifactWriter:
    """Write supplied records; simulations decide their content and timing."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory

    def append(self, name: str, record: object) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")

    def write(self, name: str, record: BaseModel | dict[str, object]) -> None:
        text = (
            record.model_dump_json(indent=2)
            if isinstance(record, BaseModel)
            else json.dumps(record, indent=2, allow_nan=False)
        )
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)

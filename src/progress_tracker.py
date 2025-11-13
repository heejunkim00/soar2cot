"""
진행 상황 추적 모듈

progress.json 파일로 SOAR 데이터 처리 진행 상황을 추적합니다.
5GB SOAR 파일을 매번 읽지 않도록 작은 JSON 파일로 관리합니다.
"""

import json
import os
from pathlib import Path
from typing import Set, Tuple

from src.log import log


# Allow custom progress file via environment variable
DEFAULT_PROGRESS_PATH = Path(
    os.getenv("PROGRESS_FILE", "/data/hjkim/soar2cot/data/progress.json")
)


class ProgressTracker:
    """SOAR 데이터 처리 진행 상황 추적"""

    def __init__(self, progress_path: Path = DEFAULT_PROGRESS_PATH):
        self.progress_path = progress_path
        self.data = self._load()
        self.completed_count_since_save = 0

    def _load(self) -> dict:
        """progress.json 로드"""
        if not self.progress_path.exists():
            log.info("No progress file found, starting fresh")
            return {
                "completed": [],
                "last_saved_to_parquet": 0,
                "total_completed": 0,
            }

        with open(self.progress_path, "r") as f:
            data = json.load(f)
            log.info(
                "Progress loaded",
                total_completed=data.get("total_completed", len(data.get("completed", []))),
                path=str(self.progress_path),
            )
            return data

    def save(self):
        """progress.json 저장"""
        with open(self.progress_path, "w") as f:
            json.dump(self.data, f, indent=2)
        log.debug("Progress saved", path=str(self.progress_path))

    def get_completed_set(self) -> Set[Tuple[str, int, str]]:
        """
        완료된 (task_id, round_index, data_type) 조합을 Set으로 반환

        Returns:
            Set of (task_id, round_index, data_type) tuples
        """
        return {
            (item["task_id"], item["round_index"], item["data_type"])
            for item in self.data.get("completed", [])
        }

    def add_completed(self, task_id: str, round_index: int, data_type: str):
        """
        완료된 샘플 추가

        Args:
            task_id: ARC task ID
            round_index: Round 번호
            data_type: 'original' 또는 'hindsight'
        """
        self.data["completed"].append(
            {
                "task_id": task_id,
                "round_index": round_index,
                "data_type": data_type,
            }
        )
        self.data["total_completed"] = len(self.data["completed"])
        self.completed_count_since_save += 1

        # 즉시 저장 (작은 파일이므로 빠름)
        self.save()

        log.debug(
            "Sample marked as completed",
            task_id=task_id,
            round_index=round_index,
            data_type=data_type,
            total_completed=self.data["total_completed"],
        )

    def should_update_parquet(self, interval: int = 5000) -> bool:
        """
        SOAR parquet 파일을 업데이트할 시점인지 확인

        Args:
            interval: 몇 개마다 업데이트할지 (기본 5000)

        Returns:
            True if should update
        """
        if self.completed_count_since_save >= interval:
            return True
        return False

    def mark_parquet_updated(self):
        """SOAR parquet 업데이트 완료 표시"""
        self.data["last_saved_to_parquet"] = self.data["total_completed"]
        self.completed_count_since_save = 0
        self.save()
        log.info(
            "Parquet update marked",
            total_completed=self.data["total_completed"],
        )

    def get_stats(self) -> dict:
        """진행 상황 통계"""
        return {
            "total_completed": self.data.get("total_completed", 0),
            "since_last_parquet_update": self.completed_count_since_save,
            "last_saved_to_parquet": self.data.get("last_saved_to_parquet", 0),
        }

"""
SOAR 데이터 라벨링 스크립트

SOAR 데이터에 다음 컬럼들을 추가:
- data_type: 'original' 또는 'hindsight'
- round_index: 각 task 내에서의 순서 (0, 1, 2, ...)
- processed: 처리 완료 여부 (False로 초기화)
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def label_soar_data():
    """SOAR 데이터에 라벨링 추가"""

    # 입력/출력 경로
    input_path = Path("/data/hjkim/soar2cot/data/soar_arc_train_5M.parquet")
    output_path = Path("/data/hjkim/soar2cot/data/soar_arc_train_5M_labeled.parquet")

    print(f"Loading SOAR data from {input_path}")

    # 데이터 로드
    table = pq.read_table(input_path)
    df = table.to_pandas()

    print(f"SOAR data loaded: {len(df)} rows, columns: {list(df.columns)}")

    # 1. data_type 추가: correct_test_input이 모두 True면 'original', 아니면 'hindsight'
    print("Adding data_type column...")
    df['data_type'] = df['correct_test_input'].apply(
        lambda x: 'original' if all(x) else 'hindsight'
    )

    original_count = (df['data_type'] == 'original').sum()
    hindsight_count = (df['data_type'] == 'hindsight').sum()

    print(f"  Original: {original_count} ({original_count/len(df)*100:.2f}%)")
    print(f"  Hindsight: {hindsight_count} ({hindsight_count/len(df)*100:.2f}%)")

    # 2. round_index 추가: 각 task_id 내에서 순서대로 번호 매기기
    print("Adding round_index column...")
    df['round_index'] = df.groupby('task_id').cumcount()

    # 통계
    max_rounds_per_task = df.groupby('task_id')['round_index'].max()
    print(f"  Min rounds per task: {int(max_rounds_per_task.min())}")
    print(f"  Max rounds per task: {int(max_rounds_per_task.max())}")
    print(f"  Avg rounds per task: {max_rounds_per_task.mean():.1f}")

    # 3. 저장 (processed 컬럼은 제거 - progress.json으로 추적)
    print(f"Saving labeled data to {output_path}")
    df.to_parquet(output_path, index=False)

    # 최종 확인
    saved_df = pd.read_parquet(output_path)
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Labeled data saved successfully!")
    print(f"  Rows: {len(saved_df)}")
    print(f"  Columns: {list(saved_df.columns)}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # 샘플 출력
    print("\n=== Sample Data ===")
    print(saved_df[['task_id', 'model', 'generation', 'data_type', 'round_index']].head(10))

    # Task별 통계
    print("\n=== Statistics by Task ===")
    task_stats = saved_df.groupby('task_id').agg({
        'round_index': 'max',
        'data_type': lambda x: f"O:{(x=='original').sum()}, H:{(x=='hindsight').sum()}"
    }).head(5)
    print(task_stats)


if __name__ == "__main__":
    label_soar_data()

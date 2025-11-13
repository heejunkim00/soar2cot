#!/usr/bin/env python3
"""
Extract perfect instructions from database.

This script extracts instructions that:
1. Achieved 100% training score (i.score = 1.0)
2. Achieved at least one correct test guess (MAX(g.avg_score) > 0)

Output files:
- perfect_instructions.csv: All data in CSV format
- perfect_instructions.json: All data in JSON format
- perfect_instructions_stats.txt: Statistics summary
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

import psycopg2
import pandas as pd
from dotenv import load_dotenv


def load_database_url():
    """Load database URL from environment."""
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    database_url = os.getenv("NEON_DSN")
    if not database_url:
        raise ValueError("NEON_DSN not found in environment variables")

    return database_url


def extract_perfect_instructions(output_dir: Path):
    """Extract perfect instructions from database and save to files."""

    # Connect to database
    database_url = load_database_url()
    print(f"Connecting to database...")
    conn = psycopg2.connect(database_url)

    try:
        # SQL query
        query = """
        SELECT
            i.task_id,
            i.soar_code as python_code,
            i.instructions,
            i.score as training_score,
            MAX(g.avg_score) as best_test_score,
            i.soar_source_model,
            i.soar_generation,
            i.soar_round_index,
            i.is_hindsight,
            COUNT(g.id) as num_guesses,
            i.created_at
        FROM instructions i
        JOIN guess g ON i.id = g.instructions_score_id
        WHERE i.score = 1.0                        -- Training 100%
        GROUP BY
            i.id,
            i.task_id,
            i.soar_code,
            i.instructions,
            i.score,
            i.soar_source_model,
            i.soar_generation,
            i.soar_round_index,
            i.is_hindsight,
            i.created_at
        HAVING MAX(g.avg_score) > 0                -- At least one test correct
        ORDER BY best_test_score DESC, i.created_at DESC
        """

        print("Executing query...")
        df = pd.read_sql(query, conn)

        if len(df) == 0:
            print("\n⚠️  No data found!")
            print("This means:")
            print("  1. Pipeline hasn't generated any perfect training scores yet")
            print("  2. Or guess table is empty (tests haven't been evaluated)")
            return

        # Print statistics
        print(f"\n{'='*60}")
        print(f"EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Total perfect instructions: {len(df)}")
        print(f"Unique tasks: {df['task_id'].nunique()}")
        print(f"")

        # Score distribution
        print("Test Score Distribution:")
        print(f"  Perfect (1.0): {len(df[df['best_test_score'] == 1.0])}")
        print(f"  Good (0.7-0.99): {len(df[(df['best_test_score'] >= 0.7) & (df['best_test_score'] < 1.0)])}")
        print(f"  Fair (0.5-0.69): {len(df[(df['best_test_score'] >= 0.5) & (df['best_test_score'] < 0.7)])}")
        print(f"  Partial (0.0-0.49): {len(df[df['best_test_score'] < 0.5])}")
        print(f"")

        # Hindsight distribution
        print("Hindsight Distribution:")
        print(f"  Original (False): {len(df[df['is_hindsight'] == False])}")
        print(f"  Hindsight (True): {len(df[df['is_hindsight'] == True])}")
        print(f"")

        # Model distribution
        if 'soar_source_model' in df.columns:
            print("Source Model Distribution:")
            for model, count in df['soar_source_model'].value_counts().items():
                print(f"  {model}: {count}")
            print(f"")

        # Save to CSV
        csv_path = output_dir / "perfect_instructions.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")

        # JSON generation disabled per user request
        # json_path = output_dir / "perfect_instructions.json"
        # df.to_json(json_path, orient='records', indent=2)
        # print(f"✅ Saved JSON: {json_path}")

        # Save statistics
        stats_path = output_dir / "perfect_instructions_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Perfect Instructions Extraction Statistics\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Unique Tasks: {df['task_id'].nunique()}\n\n")

            f.write(f"Test Score Distribution:\n")
            f.write(f"  Perfect (1.0): {len(df[df['best_test_score'] == 1.0])}\n")
            f.write(f"  Good (0.7-0.99): {len(df[(df['best_test_score'] >= 0.7) & (df['best_test_score'] < 1.0)])}\n")
            f.write(f"  Fair (0.5-0.69): {len(df[(df['best_test_score'] >= 0.5) & (df['best_test_score'] < 0.7)])}\n")
            f.write(f"  Partial (0.0-0.49): {len(df[df['best_test_score'] < 0.5])}\n\n")

            f.write(f"Hindsight Distribution:\n")
            f.write(f"  Original (False): {len(df[df['is_hindsight'] == False])}\n")
            f.write(f"  Hindsight (True): {len(df[df['is_hindsight'] == True])}\n\n")

            f.write(f"Top 10 Tasks by Test Score:\n")
            for idx, row in df.head(10).iterrows():
                f.write(f"  {row['task_id']}: train={row['training_score']:.3f}, test={row['best_test_score']:.3f}\n")

        print(f"✅ Saved Stats: {stats_path}")

        # Print sample data
        print(f"\n{'='*60}")
        print(f"SAMPLE DATA (Top 5)")
        print(f"{'='*60}")
        print(df[['task_id', 'training_score', 'best_test_score', 'is_hindsight']].head(5).to_string(index=False))

        print(f"\n{'='*60}")
        print(f"✅ Extraction complete!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


def main():
    """Main function."""
    # Output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"PERFECT INSTRUCTIONS EXTRACTION")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"")

    # Extract data
    extract_perfect_instructions(output_dir)


if __name__ == "__main__":
    main()

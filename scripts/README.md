# Scripts Directory

## extract_perfect_instructions.py

Extract instructions that achieved perfect training score (100%) and at least one correct test guess.

### Usage

```bash
# Run from project root
cd /data/hjkim/soar2cot
python scripts/extract_perfect_instructions.py
```

### Output Files

All files are saved to `/data/hjkim/soar2cot/data/`:

1. **perfect_instructions.csv** - All data in CSV format
2. **perfect_instructions.json** - All data in JSON format
3. **perfect_instructions_stats.txt** - Statistics summary

### Extraction Criteria

The script extracts instructions where:

1. **Training Score = 1.0** (100% correct on all training examples using leave-one-out cross-validation)
2. **Test Score > 0** (At least one of the two guesses got at least one test example correct)

### Output Columns

- `task_id`: ARC task ID
- `python_code`: SOAR Python solution code
- `instructions`: Generated natural language instruction
- `training_score`: Score on training examples (always 1.0)
- `best_test_score`: Best score among 2 guess attempts (0.0 ~ 1.0)
- `soar_source_model`: Source model that generated the Python code
- `soar_generation`: SOAR generation number
- `soar_round_index`: Round index in SOAR data
- `is_hindsight`: Whether this is hindsight data (True/False)
- `num_guesses`: Number of guess attempts (usually 2)
- `created_at`: Timestamp when instruction was created

### Example Output

```
task_id  | training_score | best_test_score | is_hindsight
---------|----------------|-----------------|-------------
007bbfb7 | 1.0            | 1.0             | false
05f2a901 | 1.0            | 0.667           | false
abc123   | 1.0            | 0.333           | true
```

### Requirements

- Python 3.8+
- psycopg2
- pandas
- python-dotenv

Install dependencies:
```bash
pip install psycopg2-binary pandas python-dotenv
```

### Troubleshooting

**"No data found"**:
- The pipeline hasn't generated any perfect training scores yet
- Or the guess table is empty (tests haven't been evaluated)
- Solution: Run the pipeline longer

**"NEON_DSN not found"**:
- Make sure `.env` file exists in project root
- Ensure it contains `NEON_DSN=postgresql://...`

**Connection errors**:
- Check if PostgreSQL is running
- Verify connection string in `.env`

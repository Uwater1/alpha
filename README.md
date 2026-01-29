# Alpha191 CSI 800

## Current Files
- `alpha191_cleaned.txt` : Main essay
- `alpha191.md` : Main Alpha (to be implemented)
- `bao/` : All the data (.csv)

## Alpha191 Implementation
- `alpha191/` : Factor implementations with operators and tests
- `tests/` : Unit tests for all factors
- `scripts/` : Validation scripts for running factors on real data

### Implemented Factors
- **alpha_001**: Correlation between ranked volume changes and ranked price returns (window=6)

### Usage
```bash
# Run tests
pytest tests/

# Validate alpha_001 on HS300 data
python scripts/run_alpha001.py
```

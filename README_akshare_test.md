# Akshare Test Suite

A comprehensive test script for the akshare library that discovers and tests all available functions with detailed output including real data samples and error logs.

## Features

- **Automatic Discovery**: Automatically discovers all callable functions in the akshare package
- **Smart Parameter Testing**: Intelligently tries common parameters for data functions
- **Comprehensive Coverage**: Tests functions across all akshare modules (stock, futures, fund, bond, etc.)
- **Detailed Data Output**: Shows real data samples including DataFrame shapes, columns, and actual data rows
- **Error Logging**: Provides detailed error messages and full tracebacks for failed functions
- **Progress Tracking**: Shows real-time progress during testing
- **Detailed Reports**: Generates comprehensive test result files with pass/fail/skip statistics and full details

## Requirements

- Python 3.7+
- akshare library installed
- pandas library (for data handling)
- Virtual environment (recommended)

## Installation

1. Ensure akshare is installed in your virtual environment:
```bash
pip install akshare pandas
```

## Usage

### Quick Test (5 functions per module) with data samples
```bash
python test_akshare_features.py --quick
```

### Quick Test without data samples (faster)
```bash
python test_akshare_features.py --quick --no-data
```

### Custom Number of Tests with verbose output
```bash
python test_akshare_features.py --max-tests 10 --verbose
```

### Full Test Suite (all functions) with data samples
```bash
python test_akshare_features.py
```

### Full Test Suite with verbose error logging
```bash
python test_akshare_features.py --verbose
```

### Using Virtual Environment
```bash
./alpha_env/bin/python test_akshare_features.py --quick
```

## Command Line Options

- `--quick`: Run quick test with 5 functions per module
- `--max-tests N`: Test maximum N functions per module
- `--verbose`: Show verbose output including full error tracebacks
- `--no-data`: Do not show data samples for successful tests (faster execution)

## Test Results

The script generates a detailed test result file named:
```
akshare_test_results_YYYYMMDD_HHMMSS.txt
```

The result file includes:
- Test date and time
- Total functions tested
- Pass/Fail/Skip statistics
- **Detailed success information** including:
  - Module and function name
  - Parameters used
  - Result type (DataFrame, Series, dict, etc.)
  - **Real data samples** with actual values
- **Detailed error information** for failed functions including:
  - Module and function name
  - Parameters used
  - Error message
  - Full traceback

## Example Output

### Console Output
```
================================================================================
AKSHARE COMPREHENSIVE TEST SUITE
================================================================================
Started at: 2026-01-28 22:56:26
Verbose mode: False
Show data samples: True

Discovering akshare functions...
Discovered 1054 functions across 1 modules

================================================================================
Testing module: akshare
Functions to test: 1054
================================================================================

[1/3] Testing air_city_table...
  ✓ PASSED
  Result type: DataFrame
  Shape: (168, 7)
  Columns: ['序号', '省份', '城市', 'AQI', '空气质量', 'PM2.5浓度', '首要污染物']

  First 3 rows:
    0: {'序号': 1, '省份': '北京', '城市': '北京', 'AQI': 204.0, '空气质量': '重度污染', 'PM2.5浓度': '108 ug/m3', '首要污染物': 'O3'}
    1: {'序号': 2, '省份': '河北', '城市': '廊坊', 'AQI': 199.0, '空气质量': '中度污染', 'PM2.5浓度': '54 ug/m3', '首要污染物': 'O3'}
    2: {'序号': 3, '省份': '河北', '城市': '承德', 'AQI': 198.0, '空气质量': '中度污染', 'PM2.5浓度': '59 ug/m3', '首要污染物': 'O3'}

[2/3] Testing air_quality_hebei...
  ✓ PASSED
  Result type: DataFrame
  Shape: (346, 23)
  Columns: ['城市', '区域', '监测点', '时间', 'AQI', '空气质量等级', '首要污染物', '经度', '纬度', 'PM10_IAQI']
    ... and 13 more columns

  First 3 rows:
    0: {'城市': '石家庄市', '区域': '长安区', '监测点': '和平路363号', '时间': '01/29 11:00', 'AQI': nan, '空气质量等级': '--', '首要污染物': 'e', '经度': 114.559, '纬度': 38.058, ...}

================================================================================
TEST SUMMARY
================================================================================
Total functions tested: 3
Passed: 3 (100.0%)
Failed: 0 (0.0%)
Skipped: 0 (0.0%)
Duration: 0:00:04.100162
Started: 2026-01-28 22:56:26
Ended: 2026-01-28 22:56:30

================================================================================
SUCCESSFUL FUNCTIONS SAMPLE (First 10)
================================================================================

Module: akshare
Function: air_city_table
Parameters: None
Result type: DataFrame
Result sample:
DataFrame with shape: (168, 7)
Columns: ['序号', '省份', '城市', 'AQI', '空气质量', 'PM2.5浓度', '首要污染物']

First 5 rows:
   序号  省份  城市    AQI  空气质量    PM2.5浓度 首要污染物
0   1  北京  北京  204.0  重度污染  108 ug/m3    O3
1   2  河北  廊坊  199.0  中度污染   54 ug/m3    O3
2   3  河北  承德  198.0  中度污染   59 ug/m3    O3
3   4  河北  唐山  176.0  中度污染   74 ug/m3    O3
4   5  山西  晋城  164.0  中度污染   51 ug/m3    O3

================================================================================

Test results saved to: akshare_test_results_20260128_225630.txt
```

## How It Works

1. **Discovery Phase**: The script imports akshare and discovers all callable functions
2. **Testing Phase**: For each function, it:
   - Analyzes the function signature
   - Tries calling with no parameters (if possible)
   - Tries calling with common parameters (symbol, code, date, etc.)
   - Records success, failure, or skip status
   - **Captures and displays real data samples** for successful tests
   - **Logs detailed error information** for failed tests
3. **Reporting Phase**: Generates a comprehensive test report with full details

## Tested Modules

The script tests functions from the following akshare modules:
- air (Air quality data)
- article (Article data)
- bank (Bank data)
- bond (Bond data)
- cal (Calendar utilities)
- crypto (Cryptocurrency data)
- currency (Currency data)
- data (General data)
- economic (Economic indicators)
- energy (Energy data)
- event (Event data)
- forex (Forex data)
- fortune (Fortune data)
- fund (Fund data)
- futures (Futures data)
- futures_derivative (Futures derivatives)
- fx (FX data)
- hf (High-frequency data)
- index (Index data)
- interest_rate (Interest rate data)
- movie (Movie data)
- news (News data)
- nlp (NLP data)
- option (Option data)
- other (Other data)
- pro (Professional data)
- qdii (QDII data)
- qhkc (QHKC data)
- qhkc_web (QHKC web data)
- rate (Rate data)
- reits (REITs data)
- spot (Spot data)
- stock (Stock data)
- stock_feature (Stock features)
- stock_fundamental (Stock fundamentals)
- tool (Tools)

## Output Details

### Successful Tests
For each successful test, the script displays:
- Function name and module
- Result type (DataFrame, Series, dict, list, etc.)
- For DataFrames: shape, column names, and first 3 rows with actual data
- For Series: length, index, and first 5 values
- For dicts: number of keys and sample key-value pairs
- For lists/tuples: length and first 5 items

### Failed Tests
For each failed test, the script displays:
- Function name and module
- Parameters used (if any)
- Error message
- Full traceback (in verbose mode or in result file)

## Notes

- Some functions may be skipped if they require specific parameters that cannot be automatically determined
- Network-dependent functions may fail due to connectivity issues
- The script respects rate limits and handles timeouts gracefully
- Test duration depends on the number of functions tested and network conditions
- Use `--no-data` flag for faster execution when you don't need to see data samples
- Use `--verbose` flag for detailed error tracebacks in console output

## Troubleshooting

### ModuleNotFoundError: No module named 'akshare'
Ensure akshare is installed:
```bash
pip install akshare
```

### ModuleNotFoundError: No module named 'pandas'
Ensure pandas is installed:
```bash
pip install pandas
```

### Connection Errors
Check your internet connection and try again later.

### Timeout Errors
Some functions may take longer to respond. The script handles timeouts gracefully.

### Too Much Output
Use the `--no-data` flag to suppress data samples and speed up execution:
```bash
python test_akshare_features.py --quick --no-data
```

## License

This test script is provided as-is for testing akshare library functionality.

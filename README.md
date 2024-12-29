# Pritchett Analyzer

A Python script implementing Lant Pritchett's four-part "smell tests" of development importance. Analyzes World Bank indicators against criteria for development relevance.

## Requirements

```bash
# Install dependencies
pip install pandas numpy requests click matplotlib
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kennethtiong/pritchett-analyzer.git
cd pritchett-analyzer

# Run the analysis
python pritchett_analyzer.py run-analysis

# With options
python pritchett_analyzer.py run-analysis --exclude-singapore --outdir ~/analysis --n-threads 8
```

## Python API

You can also use it in your Python code:

```python
from pritchett_analyzer import PritchettAnalyzer

analyzer = PritchettAnalyzer()
results = analyzer.run_comprehensive_analysis()
```

## License

MIT License - see LICENSE file

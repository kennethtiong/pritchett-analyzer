# Pritchett Analyzer

A simple Python script implementing Lant Pritchett's four-part "smell tests" of development importance. Analyzes World Bank indicators against criteria for development relevance.

## Install

```bash
uv pip install pritchett-analyzer
```

## Usage

```bash
# Basic usage
pritchett-analyzer run-analysis

# With options
pritchett-analyzer run-analysis --exclude-singapore --outdir ~/analysis --n-threads 8
```

Or in Python:

```python
from pritchett_analyzer import PritchettAnalyzer

analyzer = PritchettAnalyzer()
results = analyzer.run_comprehensive_analysis()
```

## License

MIT License - see LICENSE file
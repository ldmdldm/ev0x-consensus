# Consensus Benchmark

A toolkit for evaluating and comparing consensus-based approaches against single model inferences for Large Language Models (LLMs).

## Overview

This benchmark suite provides tools to quantitatively measure improvements achieved through model consensus mechanisms compared to individual model performance. It focuses on key metrics including:

- Accuracy and factual correctness
- Hallucination rates
- Response consistency
- Reasoning quality
- Performance and efficiency

## Use Cases

The benchmark is designed to demonstrate scenarios where consensus learning provides superior results:

1. **Factual Question Answering**: Reducing hallucinations and improving accuracy
2. **Reasoning Tasks**: Enhancing logical consistency and step-by-step problem solving
3. **Medical and Scientific Domains**: Providing more reliable information in critical contexts
4. **Multi-step Planning**: Improving coherence and feasibility of complex plans

## Setup

### Prerequisites

- Python 3.8+
- Access to LLM APIs (OpenAI, Claude, etc.)

### Installation

```bash
git clone https://github.com/your-username/consensus_benchmark.git
cd consensus_benchmark
pip install -r requirements.txt
```

### Configuration

1. Set up your API keys in a `.env` file:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

2. Configure model settings in `config.json` (optional)

## Running Benchmarks

```bash
# Run the full benchmark suite
python benchmark.py

# Run specific benchmark categories
python benchmark.py --category factual_qa

# Compare specific models
python benchmark.py --models gpt-4-turbo,claude-3-opus
```

## Output

Results are saved to the `results/` directory with:
- JSON files with raw metrics
- CSV exports for easy analysis
- Visualization scripts for comparative analysis

## License

MIT


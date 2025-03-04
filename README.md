# ev0x: Evolutionary Model Consensus Mechanism

![ev0x Logo](docs/images/logo.png)

## Overview

ev0x is a self-adapting AI system that runs multiple models simultaneously, compares their outputs, and evolves to improve decision making through consensus. It implements a novel approach to AI model orchestration where multiple models compete, collaborate, and evolve to provide more reliable, accurate, and unbiased results than any single model could achieve.

The system continuously evaluates model performance, distributes rewards using Shapley values, and adapts its weighting mechanisms to optimize for accuracy and diversity of thought.

This comprehensive evolutionary AI system creates a "Model DNA" profile for each AI model, tracks its performance evolution over time, and uses genetic algorithm principles to evolve the model population:
- Models with strong performance "reproduce" by creating variants
- Underperforming models are "retired" from the active population
- "Mutations" introduce novel approaches to maintain diversity

## Key Features

- **Evolutionary Model Consensus**: Self-adapting AI models that compete and evolve
- **Adaptive Model Selection**: AI models automatically ranked and selected based on performance
- **Meta-Intelligence Layer**: Creates meta-models that learn how different AI models think
- **Bias Neutralization Protocol**: Real-time bias detection and correction across models
- **Transparent Performance Tracking**: Complete visibility into model performance metrics
- **Fair Reward Distribution**: Uses Shapley values to calculate fair rewards for model contributions
- **TEE Security**: Runs in Trusted Execution Environments for data privacy and integrity

## Technical Architecture

![ev0x Architecture](docs/images/architecture.png)

### System Components

The ev0x system consists of several interconnected components that work together to provide evolutionary model consensus:

1. **Model Runner (`src/models/model_runner.py`)**
- Executes multiple AI models asynchronously
- Handles error cases and timeouts
- Provides uniform interface for different model types
- Supports batched execution for efficiency

2. **Consensus Synthesizer (`src/consensus/synthesizer.py`)**
- Compares and synthesizes outputs from multiple models
- Implements various consensus strategies (voting, weighted average, etc.)
- Handles different types of outputs (text, classifications, numerical values)

3. **Reward Calculator (`src/rewards/shapley.py`)**
- Implements Shapley value calculations for fair reward distribution
- Evaluates model contributions to final decisions
- Provides performance metrics and rankings

4. **Evolution Engine (`src/evolution/`)**
- **Selection Manager (`selection.py`)**: Handles adaptive model selection
- **Meta Intelligence (`meta_intelligence.py`)**: Creates meta-models

5. **Bias Management (`src/bias/`)**
- **Detector (`detector.py`)**: Identifies potential biases in model outputs
- **Neutralizer (`neutralizer.py`)**: Applies corrections for detected biases

6. **TEE Integration (`src/tee/`)**
- Provides secure execution environment
- Handles attestation and verification
- Ensures data privacy and model integrity

### Evolutionary Model Consensus Mechanism

The system runs multiple AI models in parallel and synthesizes their outputs using a dynamic weighting system based on:
- Historical accuracy
- Prediction confidence
- Novel solution generation
- Bias detection capabilities
- Genetic fitness scores
- Diversity contribution

#### Model DNA Structure

Each model in the ev0x ecosystem has a unique "Model DNA" profile that tracks:
- Performance characteristics across different problem domains
- Specialization patterns and strengths
- Bias tendencies and correction capabilities
- Evolution history and lineage
- Contribution diversity score

#### Genetic Algorithm Implementation

The evolutionary process follows these key steps:
1. **Evaluation**: Models are scored across multiple dimensions
2. **Selection**: Top-performing models are selected for "reproduction"
3. **Crossover**: New model variants combine strengths of parent models
4. **Mutation**: Random variations are introduced to maintain diversity
5. **Population Management**: The model population is continuously optimized

### Unique Differentiators
- First AI system that evolves its own consensus methodology
- Self-improving decision-making architecture
- Completely transparent model performance tracking
- Fair reward distribution using Shapley values
- Model DNA profiling for evolutionary model development
- Blockchain-based trust verification for model outputs
- Real-time visualization of model performance and evolution

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker (for containerized deployment)
- Access to Google Cloud Platform (for TEE deployment)
- Git

### Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ev0x.git
cd ev0x
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config/dev.json config/local.json
# Edit local.json with your specific settings
```

### Docker Installation

To build and run with Docker:

```bash
docker build -t ev0x:latest .
docker run -p 8080:8080 ev0x:latest
```

### Secure TEE Deployment

#### TEE Setup
The system operates within a Trusted Execution Environment (TEE) using Confidential Space with vTPM attestations for any of the following CPUs:
- Intel Trust Domain Extensions (TDX)
- AMD Secure Encrypted Virtualization (SEV)

#### Deployment to Google Cloud Confidential VMs

1. Set up a Google Cloud project with Confidential VM enabled
2. Authenticate with Google Cloud:
```bash
gcloud auth login
gcloud config set project your-project-id
```

3. Use the deployment script:
```bash
./ev0x-ctl.sh deploy production
```

For detailed deployment instructions, see [Deployment Guide](deployment/README.md).

## Usage

### Running the System

To run the system locally:

```bash
python main.py
```

To run specific components:

```bash
# Run the API server
python -m src.api.server

# Run benchmarks
python -m src.testing.benchmarks
```

### API Usage

The ev0x system exposes a RESTful API for integration:

#### Submit a query to multiple models and get consensus:

```bash
curl -X POST http://localhost:8080/api/v1/query \
-H "Content-Type: application/json" \
-d '{"text": "What is the capital of France?", "models": ["gemini-1.5-flash", "gemini-1.5-pro"]}'
```

#### Get model performance metrics:

```bash
curl http://localhost:8080/api/v1/models/performance
```

For full API documentation, see the [API Guide](docs/api.md).

## Data Sources and Models

### Supported AI Models

#### LLMs
- Gemini 1.5 Flash
- Gemini 1.5 Pro
- Gemini 2.0 Flash (experimental preview)

#### Specialized Models
- Text embeddings: Gemini models/text-embedding-004
- Attributed question answering: Gemini models/aqa

### Datasets
The system can leverage various datasets including:
- FTSO Block-Latency (2s latency)
- FTSO Anchor Feeds (90s latency)
- Flare Developer Hub
- GitHub Activity Data
- Google Trends Data

### Flare Ecosystem Integration
ev0x seamlessly integrates with major Flare ecosystem applications:

#### SparkDEX
- Provides optimized consensus data for DEX operations
- Processes transaction validation using multi-model verification
- Contributes to price discovery through Model DNA diversity

#### Kinetic
- Enhances NFT marketplace operations with consensus verification
- Provides tamper-proof provenance tracking within TEE environments
- Supports valuation models using diverse AI perspectives

#### Cyclo
- Powers algorithmic trading strategies with evolutionary model selection
- Provides trend analysis through consensus mechanisms
- Enhances risk modeling with multi-model perspectives

#### RainDEX
- Delivers secure oracle data through TEE-protected consensus
- Provides smart routing analysis through multi-model assessment
- Offers enhanced market analysis through evolutionary AI

### Optional Enhancements
- **Smart Contract Integration**: Compatible with Flare's TeeV1Verifier and TeeV1Interface smart contracts
- **GPU Acceleration**: Supports Confidential Accelerators with NVIDIA H100
## Development Guide

### Project Structure

```
ev0x/
├── src/                      # Source code
│   ├── api/                  # API server
│   ├── bias/                 # Bias detection and neutralization
│   ├── config/               # Configuration management
│   ├── consensus/            # Consensus generation
│   ├── data/                 # Data access and management
│   ├── evolution/            # Evolutionary components
│   ├── models/               # Model execution
│   ├── rewards/              # Reward calculation
│   ├── tee/                  # TEE integration
│   └── utils/                # Utilities
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── deployment/               # Deployment configurations
│   ├── kubernetes/           # Kubernetes manifests
│   ├── monitoring/           # Monitoring configurations
│   └── security/             # Security guides and scripts
├── config/                   # Environment configurations
├── docs/                     # Documentation
│   └── images/               # Architecture diagrams
├── .github/                  # GitHub workflows
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── ev0x-ctl.sh               # Management script
├── main.py                   # Application entry point
└── README.md                 # This file
```

### Development Workflow

1. **Set up development environment**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Run tests**:
```bash
pytest
```

3. **Format code**:
```bash
black .
```

4. **Run linting**:
```bash
flake8
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For more details, see the [Contributing Guide](docs/CONTRIBUTING.md).

## Security Considerations

### TEE Security

The ev0x system is designed to operate within a Trusted Execution Environment (TEE) to ensure:
- Data confidentiality
- Code integrity
- Attestation of execution environment

### Data Protection

- All sensitive data is processed exclusively within the TEE
- Model weights and inference are protected from unauthorized access
- Network communications are encrypted

### Attestation

The system supports remote attestation to verify:
- The integrity of the execution environment
- The authenticity of the running code
- Compliance with security policies

For more details on security best practices when deploying ev0x, see the [Security Guide](deployment/security/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Cloud for Confidential Computing infrastructure
- Flare Network for blockchain integration capabilities
- The open source community for various tools and libraries

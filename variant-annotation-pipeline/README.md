# Agentic Variant Annotation Pipeline

A robust, AI-powered variant annotation pipeline using CrewAI that automatically processes VCF files, calls multiple annotation APIs with intelligent fallback, and generates comprehensive variant reports.

## Features

- **Multi-Agent Architecture**: DataLoader, Annotator, and Reporter agents working collaboratively
- **Multiple API Support**: MyVariant.info, Ensembl VEP, ClinVar with intelligent fallback
- **LLM-Powered Intelligence**: Uses GPT-4, Claude, Gemini, or local models for intelligent decision making
- **YAML Configuration**: Easy configuration management through YAML files
- **Error Resilience**: Automatic retry logic and graceful degradation
- **Flexible Input**: Supports various VCF formats and sizes
- **Rich Reporting**: Generates detailed HTML and CSV reports
- **Configurable**: Multiple configuration options (YAML, environment variables)

## Prerequisites

- Python 3.8+
- API key for your chosen LLM provider (OpenAI, Anthropic, Google Gemini, Azure, or local Ollama)
- Internet connection for annotation APIs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your LLM

Choose one of the following configuration methods:

#### Option A: YAML Configuration (Recommended)

Copy the YAML template and edit it:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys and preferences
```

Example `config.yaml` for Gemini:
```yaml
llm:
  provider: "gemini"
  model: "gemini-pro"
  temperature: 0.1
  max_tokens: 4000

api_keys:
  google: "your-google-api-key-here"
```

#### Option B: Environment Variables

**Using Google Gemini (Recommended - Fast and Cost-Effective)**
```bash
export GOOGLE_API_KEY="your-google-api-key"
export LLM_PROVIDER="gemini"
export LLM_MODEL="gemini-pro"
```

**Using OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4"
```

**Using Anthropic/Claude**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-sonnet-20240229"
```

**Using Azure OpenAI**
```bash
export AZURE_OPENAI_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export LLM_PROVIDER="azure"
export LLM_MODEL="gpt-4"
```

**Using Local Ollama (Free)**
```bash
# First install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama2

# Then configure
export LLM_PROVIDER="ollama"
export LLM_MODEL="llama2"
```

### 3. Run the Pipeline

```bash
python main.py --input data/input/sample.vcf --output data/output/
```

## Configuration Options

### YAML Configuration (Preferred)

The pipeline supports comprehensive YAML configuration. Create a `config.yaml` file:

```yaml
# LLM Configuration
llm:
  provider: "gemini"  # openai, anthropic, azure, gemini, ollama
  model: "gemini-pro"
  temperature: 0.1
  max_tokens: 4000

# API Keys
api_keys:
  google: "your-google-api-key"
  openai: "your-openai-api-key"
  anthropic: "your-anthropic-api-key"

# Pipeline Settings
pipeline:
  max_variants_per_batch: 100
  min_quality_score: 20.0
  output_formats: ["csv", "xlsx", "html"]
  log_level: "INFO"

# Performance
performance:
  max_concurrent_requests: 5
  cache_annotations: true
```

### Environment Variables

All YAML settings can be overridden with environment variables:

```bash
# LLM Configuration
LLM_PROVIDER=gemini                    # openai, anthropic, azure, gemini, ollama
LLM_MODEL=gemini-pro                   # Model name
LLM_TEMPERATURE=0.1                   # Creativity (0.0-1.0)
LLM_MAX_TOKENS=4000                   # Max response length

# Pipeline Settings
MAX_VARIANTS_PER_BATCH=100            # Batch size for processing
MIN_QUALITY_SCORE=20.0                # Minimum variant quality
LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR

# Performance
MAX_CONCURRENT_REQUESTS=5             # Concurrent API calls
CACHE_ANNOTATIONS=true                # Enable caching
```

### LLM Provider Comparison

| Provider | Cost | Speed | Setup | Recommended For |
|----------|------|-------|-------|-----------------|
| **Gemini** | $ | Fast | Easy | Best balance of cost/performance |
| **OpenAI** | $$$ | Fast | Easy | Highest quality results |
| **Anthropic** | $$ | Fast | Easy | Alternative to OpenAI |
| **Azure** | $$$ | Fast | Medium | Enterprise environments |
| **Ollama** | Free | Slow | Medium | Local development, privacy |

## Usage Examples

### Basic Usage
```bash
python main.py --input my_variants.vcf
```

### Using Custom Configuration
```bash
python main.py --input my_variants.vcf --config custom_config.yaml
```

### Advanced Usage
```bash
python main.py \
  --input large_dataset.vcf \
  --output /path/to/results/ \
  --formats csv xlsx html \
  --min-quality 30 \
  --max-variants 1000 \
  --verbose
```

### Batch Processing
```bash
# Process multiple files
for vcf in *.vcf; do
    python main.py --input "$vcf" --output "results/${vcf%.vcf}/"
done
```

## Getting API Keys

### Google Gemini (Recommended)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your configuration

### OpenAI
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key to your configuration

### Anthropic
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Generate an API key
3. Copy the key to your configuration

## Project Structure

```
variant-annotation-pipeline/
├── config/
│   ├── __init__.py
│   ├── settings.py         # Main configuration with YAML support
│   └── agents_config.py    # Agent-specific settings with Gemini support
├── utils/
│   ├── __init__.py
│   ├── vcf_parser.py       # VCF file parsing
│   ├── api_clients.py      # API client management
│   └── data_handlers.py    # Data processing utilities
├── agents/
│   ├── __init__.py
│   ├── data_loader_agent.py
│   ├── annotator_agent.py
│   └── reporter_agent.py
├── tools/
│   ├── __init__.py
│   ├── vcf_tools.py        # VCF validation tools
│   └── annotation_tools.py # Annotation retry tools
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies (includes Gemini support)
├── config.yaml.example     # YAML configuration template
├── environment.example     # Environment template
└── README.md              # Documentation
```

## How It Works

### 1. **Data Loading Agent** 🔍
- Validates VCF file format
- Applies quality filters
- Extracts variant information
- Generates data quality statistics

### 2. **Annotator Agent** 🔬
- Queries multiple annotation APIs
- Implements intelligent fallback strategies
- Handles rate limiting and retries
- Merges annotation data

### 3. **Reporter Agent** 📊
- Generates comprehensive reports
- Creates visualizations
- Provides actionable recommendations
- Outputs in multiple formats

## Output Files

The pipeline generates several output files:

- **CSV**: Machine-readable variant data
- **Excel**: Multi-sheet report with summary
- **HTML**: Interactive web report
- **Summary**: Human-readable analysis summary

## Troubleshooting

### Common Issues

**1. LLM API Key Not Set**
```
⚠️ Warning: No API key found for gemini.
```
**Solution**: Set your API key in YAML config or environment: `export GOOGLE_API_KEY="your-key"`

**2. YAML Configuration Error**
```
Error loading YAML config: ...
```
**Solution**: Check YAML syntax, ensure proper indentation

**3. Rate Limiting**
```
Rate limited by myvariant. Waiting 5s
```
**Solution**: Normal behavior - the pipeline automatically handles rate limits

**4. No Variants Found**
```
Error: No variants found in VCF file
```
**Solution**: Check your VCF file format and quality thresholds

### Getting Help

1. Check the logs in `variant_annotation.log`
2. Enable verbose logging: `--verbose`
3. Test with a smaller dataset: `--max-variants 10`
4. Validate your configuration: Check `config.yaml` syntax

## Performance Considerations

- **Small datasets** (< 1000 variants): Run on any machine
- **Medium datasets** (1000-10000 variants): Consider cloud instance
- **Large datasets** (> 10000 variants): Use distributed processing

## API Costs

Approximate costs for 1000 variants (depends on annotation complexity):

- **Google Gemini**: ~$0.10-0.30 💰 *Most cost-effective*
- **OpenAI GPT-4**: ~$0.50-2.00
- **Anthropic Claude**: ~$0.40-1.50
- **Azure OpenAI**: Similar to OpenAI
- **Ollama**: Free (local processing)

## Configuration Priority

The pipeline loads configuration in this order (later sources override earlier ones):

1. Default values
2. YAML configuration file (`config.yaml`)
3. Environment variables
4. Command-line arguments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details. 
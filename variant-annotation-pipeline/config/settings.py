"""
Main configuration settings for the variant annotation pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for annotation APIs."""
    name: str
    base_url: str
    rate_limit: int  # requests per second
    timeout: int     # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds

@dataclass
class LLMConfig:
    """Configuration for Language Model settings."""
    provider: str  # "openai", "anthropic", "azure", "gemini", "ollama", etc.
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    max_retries: int = 3

@dataclass
class Settings:
    """Main settings for the variant annotation pipeline."""
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    
    # LLM Configuration
    llm_config: LLMConfig = field(default_factory=lambda: LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        model=os.getenv("LLM_MODEL", "gemini-pro"),
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY"),
        api_base=os.getenv("LLM_API_BASE"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000"))
    ))
    
    # Alternative LLM configurations for different providers
    llm_providers: Dict[str, LLMConfig] = field(default_factory=lambda: {
        "openai": LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=4000
        ),
        "anthropic": LLMConfig(
            provider="anthropic", 
            model="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
            max_tokens=4000
        ),
        "gemini": LLMConfig(
            provider="gemini",
            model="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_tokens=4000
        ),
        "ollama": LLMConfig(
            provider="ollama",
            model="llama2",
            api_base="http://localhost:11434",
            temperature=0.1,
            max_tokens=4000
        ),
        "azure": LLMConfig(
            provider="azure",
            model="gpt-4",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0.1,
            max_tokens=4000
        )
    })
    
    # VCF processing
    max_variants_per_batch: int = 100
    min_quality_score: float = 20.0
    include_info_fields: List[str] = field(default_factory=lambda: [
        'DP', 'AF', 'AC', 'AN', 'QUAL'
    ])
    
    # Annotation APIs (in order of preference)
    annotation_apis: List[APIConfig] = field(default_factory=lambda: [
        APIConfig(
            name="myvariant",
            base_url="https://myvariant.info/v1",
            rate_limit=10,
            timeout=30
        ),
        APIConfig(
            name="ensembl",
            base_url="https://rest.ensembl.org",
            rate_limit=15,
            timeout=30
        ),
        APIConfig(
            name="clinvar",
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=3,
            timeout=30
        )
    ])
    
    # Annotation fields to retrieve
    annotation_fields: Dict[str, List[str]] = field(default_factory=lambda: {
        'myvariant': [
            'dbsnp.rsid',
            'cadd.phred',
            'clinvar.clinical_significance',
            'gnomad.af.af',
            'dbnsfp.sift.pred',
            'dbnsfp.polyphen2.hdiv.pred'
        ],
        'ensembl': [
            'consequence_terms',
            'impact',
            'gene_symbol',
            'transcript_id',
            'protein_id'
        ],
        'clinvar': [
            'clinical_significance',
            'review_status',
            'condition'
        ]
    })
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ['csv', 'xlsx', 'html'])
    include_plots: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "variant_annotation.log"
    
    # Performance
    max_concurrent_requests: int = 5
    cache_annotations: bool = True
    cache_ttl_hours: int = 24
    
    # CrewAI specific settings
    crew_verbose: bool = True
    crew_memory: bool = True
    crew_max_execution_time: int = 300  # 5 minutes per agent
    
    def __post_init__(self):
        """Initialize derived paths and validate configuration."""
        self.data_dir = self.project_root / "data"
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.input_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate LLM configuration
        self._validate_llm_config()
    
    def _validate_llm_config(self):
        """Validate LLM configuration and warn about missing API keys."""
        if not self.llm_config.api_key and self.llm_config.provider not in ["ollama"]:
            print(f"⚠️  Warning: No API key found for {self.llm_config.provider}.")
            print(f"   Please set the appropriate environment variable or YAML config:")
            if self.llm_config.provider == "openai":
                print("   export OPENAI_API_KEY='your-api-key'")
            elif self.llm_config.provider == "anthropic":
                print("   export ANTHROPIC_API_KEY='your-api-key'")
            elif self.llm_config.provider == "gemini":
                print("   export GOOGLE_API_KEY='your-api-key'")
            elif self.llm_config.provider == "azure":
                print("   export AZURE_OPENAI_KEY='your-api-key'")
                print("   export AZURE_OPENAI_ENDPOINT='your-endpoint'")
            else:
                print("   export LLM_API_KEY='your-api-key'")
    
    def get_llm_config(self, provider: str = None) -> LLMConfig:
        """Get LLM configuration for a specific provider."""
        if provider and provider in self.llm_providers:
            return self.llm_providers[provider]
        return self.llm_config
    
    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """Load settings from a YAML configuration file."""
        logger.info(f"Loading configuration from {yaml_path}")
        
        if not yaml_path.exists():
            logger.warning(f"YAML config file not found: {yaml_path}")
            return cls.from_env()
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            settings = cls()
            
            # Load LLM configuration
            if 'llm' in config_data:
                llm_config = config_data['llm']
                api_keys = config_data.get('api_keys', {})
                
                provider = llm_config.get('provider', 'gemini')
                model = llm_config.get('model', 'gemini-pro')
                
                # Get API key based on provider
                api_key = None
                if provider == 'openai':
                    api_key = api_keys.get('openai')
                elif provider == 'anthropic':
                    api_key = api_keys.get('anthropic')
                elif provider == 'gemini':
                    api_key = api_keys.get('google')
                elif provider == 'azure':
                    api_key = api_keys.get('azure', {}).get('key')
                    settings.llm_config.api_base = api_keys.get('azure', {}).get('endpoint')
                elif provider == 'ollama':
                    api_key = None  # Ollama doesn't need API key
                else:
                    api_key = api_keys.get('generic', {}).get('key')
                
                settings.llm_config = LLMConfig(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    api_base=llm_config.get('api_base') or settings.llm_config.api_base,
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens', 4000),
                    timeout=llm_config.get('timeout', 60),
                    max_retries=llm_config.get('max_retries', 3)
                )
            
            # Load pipeline configuration
            if 'pipeline' in config_data:
                pipeline_config = config_data['pipeline']
                settings.max_variants_per_batch = pipeline_config.get('max_variants_per_batch', 100)
                settings.min_quality_score = pipeline_config.get('min_quality_score', 20.0)
                settings.include_info_fields = pipeline_config.get('include_info_fields', settings.include_info_fields)
                settings.output_formats = pipeline_config.get('output_formats', settings.output_formats)
                settings.include_plots = pipeline_config.get('include_plots', True)
                settings.log_level = pipeline_config.get('log_level', 'INFO')
                settings.log_file = pipeline_config.get('log_file', 'variant_annotation.log')
            
            # Load performance configuration
            if 'performance' in config_data:
                perf_config = config_data['performance']
                settings.max_concurrent_requests = perf_config.get('max_concurrent_requests', 5)
                settings.cache_annotations = perf_config.get('cache_annotations', True)
                settings.cache_ttl_hours = perf_config.get('cache_ttl_hours', 24)
            
            # Load CrewAI configuration
            if 'crew' in config_data:
                crew_config = config_data['crew']
                settings.crew_verbose = crew_config.get('verbose', True)
                settings.crew_memory = crew_config.get('memory', True)
                settings.crew_max_execution_time = crew_config.get('max_execution_time', 300)
            
            # Load annotation APIs configuration
            if 'annotation_apis' in config_data:
                apis_config = config_data['annotation_apis']
                settings.annotation_apis = []
                settings.annotation_fields = {}
                
                for api_name, api_config in apis_config.items():
                    settings.annotation_apis.append(APIConfig(
                        name=api_name,
                        base_url=api_config['base_url'],
                        rate_limit=api_config.get('rate_limit', 10),
                        timeout=api_config.get('timeout', 30),
                        retry_attempts=api_config.get('retry_attempts', 3),
                        retry_delay=api_config.get('retry_delay', 1.0)
                    ))
                    
                    if 'fields' in api_config:
                        settings.annotation_fields[api_name] = api_config['fields']
            
            logger.info(f"Successfully loaded configuration from {yaml_path}")
            return settings
            
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            logger.info("Falling back to environment variables")
            return cls.from_env()
    
    @classmethod
    def from_env(cls):
        """Load settings from environment variables."""
        settings = cls()
        
        # Override with environment variables if present
        if "MAX_VARIANTS_PER_BATCH" in os.environ:
            settings.max_variants_per_batch = int(os.environ["MAX_VARIANTS_PER_BATCH"])
        
        if "MIN_QUALITY_SCORE" in os.environ:
            settings.min_quality_score = float(os.environ["MIN_QUALITY_SCORE"])
        
        if "LOG_LEVEL" in os.environ:
            settings.log_level = os.environ["LOG_LEVEL"]
        
        return settings
    
    @classmethod
    def load(cls, config_path: str = None):
        """Load settings from YAML file or environment variables."""
        if config_path:
            config_file = Path(config_path)
        else:
            # Try to find config files in common locations
            possible_configs = [
                Path("config.yaml"),
                Path("config.yml"),
                Path("config/config.yaml"),
                Path("config/config.yml")
            ]
            
            config_file = None
            for path in possible_configs:
                if path.exists():
                    config_file = path
                    break
        
        if config_file and config_file.exists():
            return cls.from_yaml(config_file)
        else:
            logger.info("No YAML config found, using environment variables")
            return cls.from_env()

# Column mapping for VCF parsing
COLUMN_MAPPING = {
    '#CHROM': 'CHROM',
    'CHROM': 'CHROM',
    'POS': 'POS',
    'ID': 'ID',
    'REF': 'REF',
    'ALT': 'ALT',
    'QUAL': 'QUAL',
    'FILTER': 'FILTER',
    'INFO': 'INFO'
}

# Valid amino acids for protein sequences
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

# Global settings instance - automatically loads from YAML or environment
settings = Settings.load() 
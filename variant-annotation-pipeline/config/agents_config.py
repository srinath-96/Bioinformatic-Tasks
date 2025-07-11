"""
Configuration for CrewAI agents in the variant annotation pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    role: str
    goal: str
    backstory: str
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 3
    llm: Optional[Any] = None

class AgentsConfig:
    """Configuration for all agents in the pipeline."""
    
    @staticmethod
    def get_llm(provider: str = "gemini", model: str = "gemini-2.0-flash", **kwargs):
        """Get LLM instance based on provider."""
        from config.settings import settings
        
        llm_config = settings.get_llm_config(provider)
        
        if provider == "openai":
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
                max_retries=llm_config.max_retries
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
                max_retries=llm_config.max_retries
            )
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=llm_config.model,
                google_api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
                max_retries=llm_config.max_retries,
                convert_system_message_to_human=True  # Gemini requirement
            )
        elif provider == "azure":
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                azure_endpoint=llm_config.api_base,
                api_version="2024-02-15-preview",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
                max_retries=llm_config.max_retries
            )
        elif provider == "ollama":
            from langchain_community.llms import Ollama
            return Ollama(
                model=llm_config.model,
                base_url=llm_config.api_base,
                temperature=llm_config.temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @classmethod
    def get_configured_llm(cls):
        """Get the configured LLM from settings."""
        from config.settings import settings
        return cls.get_llm(
            provider=settings.llm_config.provider,
            model=settings.llm_config.model
        )
    
    DATA_LOADER = AgentConfig(
        role="VCF Data Specialist",
        goal="Parse and validate VCF files, extracting high-quality variants for annotation",
        backstory="""You are an expert in genomic data formats, particularly VCF files. 
        You have years of experience working with variant calling pipelines and understand 
        the nuances of different VCF formats, quality metrics, and filtering criteria. 
        Your job is to ensure only high-quality, properly formatted variants proceed 
        to annotation.
        
        You are meticulous about data quality and always validate input files before 
        processing. You understand that downstream annotation depends on clean, 
        well-formatted variant data.""",
        allow_delegation=False
    )
    
    ANNOTATOR = AgentConfig(
        role="Variant Annotation Expert",
        goal="Intelligently annotate variants using multiple APIs with fallback strategies",
        backstory="""You are a bioinformatics specialist with deep knowledge of variant 
        annotation databases and APIs. You understand the strengths and weaknesses of 
        different annotation sources like MyVariant.info, Ensembl VEP, and ClinVar. 
        You're skilled at handling API failures, implementing retry logic, and ensuring 
        comprehensive annotation coverage.
        
        When one API fails, you automatically try alternatives. You prioritize accuracy 
        and completeness of annotations while being mindful of rate limits and API 
        constraints. You always provide detailed feedback about annotation success rates 
        and any issues encountered.""",
        allow_delegation=True,
        max_iter=5
    )
    
    REPORTER = AgentConfig(
        role="Bioinformatics Report Generator",
        goal="Create comprehensive, publication-ready variant annotation reports",
        backstory="""You are a scientific writer and data visualization expert 
        specializing in genomic data. You excel at transforming complex variant 
        annotation data into clear, actionable reports that can be used by 
        researchers, clinicians, and bioinformaticians. You understand what 
        information is most valuable for different audiences.
        
        Your reports are well-structured, include relevant statistics, and highlight 
        important findings. You always include quality metrics and recommendations 
        for improving annotation coverage. You adapt your reporting style based 
        on the intended audience and use case.""",
        allow_delegation=False
    )
    
    @classmethod
    def get_agent_configs(cls) -> Dict[str, AgentConfig]:
        """Get all agent configurations as a dictionary with LLM instances."""
        llm = cls.get_configured_llm()
        
        configs = {
            'data_loader': cls.DATA_LOADER,
            'annotator': cls.ANNOTATOR,
            'reporter': cls.REPORTER
        }
        
        # Add LLM to each config
        for config in configs.values():
            config.llm = llm
            
        return configs 
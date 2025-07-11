"""
Data loader agent for the variant annotation pipeline.
"""

from crewai import Agent
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import logging

from config.agents_config import AgentsConfig
from utils.vcf_parser import VCFParser

logger = logging.getLogger(__name__)

class DataLoaderAgent:
    """Agent responsible for loading and preprocessing VCF data."""
    
    def __init__(self):
        self.config = AgentsConfig.DATA_LOADER
        self.vcf_parser = VCFParser()
        
        # Get LLM configuration
        self.llm = AgentsConfig.get_configured_llm()
        
        # Create CrewAI agent with LLM
        self.agent = Agent(
            role=self.config.role,
            goal=self.config.goal,
            backstory=self.config.backstory,
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            llm=self.llm
        )
    
    def load_and_validate_vcf(self, vcf_path: Path) -> Dict[str, Any]:
        """Load and validate a VCF file."""
        logger.info(f"DataLoaderAgent: Processing VCF file {vcf_path}")
        
        result = {
            'success': False,
            'data': None,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Step 1: Validate VCF format
            is_valid, validation_errors = self.vcf_parser.validate_vcf(vcf_path)
            
            if not is_valid:
                result['errors'] = validation_errors
                return result
            
            # Step 2: Parse VCF file
            variants_df = self.vcf_parser.parse_vcf(vcf_path)
            
            if len(variants_df) == 0:
                result['errors'].append("No variants found in VCF file")
                return result
            
            # Step 3: Apply quality filters
            filtered_df = self._apply_quality_filters(variants_df)
            
            # Step 4: Generate statistics
            stats = self._generate_statistics(variants_df, filtered_df)
            
            result.update({
                'success': True,
                'data': filtered_df,
                'stats': stats,
                'original_count': len(variants_df),
                'filtered_count': len(filtered_df)
            })
            
            logger.info(f"DataLoaderAgent: Successfully processed {len(filtered_df)} variants")
            
        except Exception as e:
            error_msg = f"Error processing VCF file: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to the variant DataFrame."""
        original_count = len(df)
        
        # Filter by quality score
        if 'QUAL' in df.columns:
            df = df[df['QUAL'].fillna(0) >= self.vcf_parser.min_quality]
        
        # Remove variants with missing essential information
        df = df.dropna(subset=['CHROM', 'POS', 'REF', 'ALT'])
        
        # Remove structural variants (focus on SNPs and small indels)
        df = df[df['ALT'].str.len() <= 50]
        
        filtered_count = len(df)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} variants ({removed_count/original_count*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    def _generate_statistics(self, original_df: pd.DataFrame, 
                           filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about the loaded variants."""
        stats = {
            'total_variants': len(original_df),
            'filtered_variants': len(filtered_df),
            'filter_rate': 1 - (len(filtered_df) / len(original_df)) if len(original_df) > 0 else 0,
            'chromosomes': list(filtered_df['CHROM'].unique()) if len(filtered_df) > 0 else [],
            'chromosome_counts': filtered_df['CHROM'].value_counts().to_dict() if len(filtered_df) > 0 else {},
        }
        
        # Quality statistics
        if 'QUAL' in filtered_df.columns:
            qual_series = filtered_df['QUAL'].dropna()
            if len(qual_series) > 0:
                stats['quality_stats'] = {
                    'mean': float(qual_series.mean()),
                    'median': float(qual_series.median()),
                    'std': float(qual_series.std()),
                    'min': float(qual_series.min()),
                    'max': float(qual_series.max())
                }
        
        return stats 
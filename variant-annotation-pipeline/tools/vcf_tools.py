"""
VCF-specific tools for the data loader agent.
"""

from crewai_tools import BaseTool
from typing import Dict, Any
import pandas as pd
from pathlib import Path

class VCFValidationTool(BaseTool):
    """Tool for validating VCF file format and structure."""
    
    name: str = "VCF Validator"
    description: str = "Validates VCF file format, checks for required columns, and identifies potential issues."
    
    def _run(self, vcf_path: str) -> Dict[str, Any]:
        """Validate a VCF file."""
        from utils.vcf_parser import VCFParser
        
        parser = VCFParser()
        is_valid, errors = parser.validate_vcf(Path(vcf_path))
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'file_path': vcf_path
        }

class VCFFilterTool(BaseTool):
    """Tool for applying filters to VCF data."""
    
    name: str = "VCF Filter"
    description: str = "Applies quality filters and other criteria to VCF variant data."
    
    def _run(self, variants_data: str, filter_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to variant data."""
        import json
        
        # Parse the variants data
        if isinstance(variants_data, str):
            try:
                data = json.loads(variants_data)
                df = pd.DataFrame(data)
            except:
                return {'error': 'Invalid variants data format'}
        else:
            df = pd.DataFrame(variants_data)
        
        original_count = len(df)
        
        # Apply filters based on criteria
        if 'min_quality' in filter_criteria and 'QUAL' in df.columns:
            df = df[df['QUAL'] >= filter_criteria['min_quality']]
        
        if 'chromosomes' in filter_criteria:
            df = df[df['CHROM'].isin(filter_criteria['chromosomes'])]
        
        filtered_count = len(df)
        
        return {
            'filtered_data': df.to_dict('records'),
            'original_count': original_count,
            'filtered_count': filtered_count,
            'filter_rate': 1 - (filtered_count / original_count) if original_count > 0 else 0
        } 
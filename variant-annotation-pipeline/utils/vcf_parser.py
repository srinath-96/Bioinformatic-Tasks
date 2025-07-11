"""
VCF file parsing utilities.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

try:
    import cyvcf2
    CYVCF2_AVAILABLE = True
except ImportError:
    CYVCF2_AVAILABLE = False
    logging.warning("cyvcf2 not available. Falling back to basic VCF parsing.")

from config.settings import settings

logger = logging.getLogger(__name__)

class VCFParser:
    """Robust VCF file parser with quality filtering."""
    
    def __init__(self, min_quality: float = None):
        self.min_quality = min_quality or settings.min_quality_score
        self.info_fields = settings.include_info_fields
    
    def parse_vcf(self, vcf_path: Path) -> pd.DataFrame:
        """Parse VCF file and return a DataFrame of variants."""
        logger.info(f"Parsing VCF file: {vcf_path}")
        
        if CYVCF2_AVAILABLE:
            return self._parse_with_cyvcf2(vcf_path)
        else:
            return self._parse_basic(vcf_path)
    
    def _parse_with_cyvcf2(self, vcf_path: Path) -> pd.DataFrame:
        """Parse VCF using cyvcf2 (faster and more robust)."""
        variants = []
        
        try:
            vcf = cyvcf2.VCF(str(vcf_path))
            
            for i, variant in enumerate(vcf):
                # Quality filtering
                if variant.QUAL is not None and variant.QUAL < self.min_quality:
                    continue
                
                # Extract basic variant info
                var_data = {
                    'CHROM': variant.CHROM,
                    'POS': variant.POS,
                    'ID': variant.ID or '.',
                    'REF': variant.REF,
                    'ALT': ','.join(str(alt) for alt in variant.ALT),
                    'QUAL': variant.QUAL,
                    'FILTER': variant.FILTER or 'PASS'
                }
                
                # Extract INFO fields
                for field in self.info_fields:
                    try:
                        value = variant.INFO.get(field)
                        var_data[f'INFO_{field}'] = value
                    except:
                        var_data[f'INFO_{field}'] = None
                
                # Create variant identifier for annotation
                var_data['variant_id'] = self._create_variant_id(
                    variant.CHROM, variant.POS, variant.REF, variant.ALT[0]
                )
                
                variants.append(var_data)
            
            vcf.close()
            
        except Exception as e:
            logger.error(f"Error parsing VCF with cyvcf2: {e}")
            raise
        
        df = pd.DataFrame(variants)
        logger.info(f"Parsed {len(df)} variants from {vcf_path}")
        return df
    
    def _parse_basic(self, vcf_path: Path) -> pd.DataFrame:
        """Basic VCF parsing without cyvcf2."""
        variants = []
        
        try:
            with open(vcf_path, 'r') as f:
                header_line = None
                
                for line in f:
                    line = line.strip()
                    
                    # Skip header lines
                    if line.startswith('##'):
                        continue
                    
                    # Column header line
                    if line.startswith('#CHROM'):
                        header_line = line[1:].split('\t')
                        continue
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Parse variant line
                    fields = line.split('\t')
                    
                    if len(fields) < 8:
                        continue
                    
                    # Basic quality filtering
                    try:
                        qual = float(fields[5]) if fields[5] != '.' else None
                        if qual is not None and qual < self.min_quality:
                            continue
                    except:
                        pass
                    
                    # Extract variant data
                    var_data = {
                        'CHROM': fields[0],
                        'POS': int(fields[1]),
                        'ID': fields[2] if fields[2] != '.' else None,
                        'REF': fields[3],
                        'ALT': fields[4],
                        'QUAL': qual,
                        'FILTER': fields[6] if fields[6] != '.' else 'PASS'
                    }
                    
                    # Parse INFO field
                    info_data = self._parse_info_field(fields[7])
                    for field in self.info_fields:
                        var_data[f'INFO_{field}'] = info_data.get(field)
                    
                    # Create variant identifier
                    alt_allele = fields[4].split(',')[0]  # Take first ALT allele
                    var_data['variant_id'] = self._create_variant_id(
                        fields[0], int(fields[1]), fields[3], alt_allele
                    )
                    
                    variants.append(var_data)
                    
        except Exception as e:
            logger.error(f"Error parsing VCF: {e}")
            raise
        
        df = pd.DataFrame(variants)
        logger.info(f"Parsed {len(df)} variants from {vcf_path}")
        return df
    
    def _parse_info_field(self, info_str: str) -> Dict[str, Any]:
        """Parse the INFO field from a VCF line."""
        info_data = {}
        
        if info_str == '.':
            return info_data
        
        for item in info_str.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                # Try to convert to appropriate type
                try:
                    if ',' in value:
                        info_data[key] = [float(v) if '.' in v else int(v) for v in value.split(',')]
                    else:
                        info_data[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    info_data[key] = value
            else:
                info_data[item] = True
        
        return info_data
    
    def _create_variant_id(self, chrom: str, pos: int, ref: str, alt: str) -> str:
        """Create a standardized variant identifier."""
        # Remove 'chr' prefix if present for consistency
        chrom = chrom.replace('chr', '')
        return f"{chrom}:{pos}:{ref}>{alt}"
    
    def validate_vcf(self, vcf_path: Path) -> Tuple[bool, List[str]]:
        """Validate VCF file format."""
        errors = []
        
        if not vcf_path.exists():
            errors.append(f"File does not exist: {vcf_path}")
            return False, errors
        
        try:
            with open(vcf_path, 'r') as f:
                has_header = False
                has_variants = False
                line_count = 0
                
                for line in f:
                    line_count += 1
                    line = line.strip()
                    
                    if line.startswith('##fileformat=VCF'):
                        has_header = True
                    elif line.startswith('#CHROM'):
                        # Check required columns
                        cols = line[1:].split('\t')
                        required_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
                        for col in required_cols:
                            if col not in cols:
                                errors.append(f"Missing required column: {col}")
                    elif not line.startswith('#') and line:
                        has_variants = True
                        # Validate first variant line
                        fields = line.split('\t')
                        if len(fields) < 8:
                            errors.append(f"Line {line_count}: Insufficient fields (need at least 8)")
                        break
                
                if not has_header:
                    errors.append("Missing VCF header")
                if not has_variants:
                    errors.append("No variant lines found")
                    
        except Exception as e:
            errors.append(f"Error reading file: {e}")
        
        return len(errors) == 0, errors 
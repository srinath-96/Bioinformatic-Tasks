"""
Data handling utilities for the variant annotation pipeline.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
from datetime import datetime, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data operations including caching and output formatting."""
    
    def __init__(self):
        self.cache_dir = settings.project_root / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def merge_variant_annotations(self, variants_df: pd.DataFrame, 
                                 annotation_results: Dict[str, List]) -> pd.DataFrame:
        """Merge variant data with annotation results."""
        annotated_df = variants_df.copy()
        
        # Initialize annotation columns
        annotation_columns = {}
        for api_name in settings.annotation_apis:
            for field in settings.annotation_fields.get(api_name.name, []):
                col_name = f"{api_name.name}_{field.replace('.', '_')}"
                annotation_columns[col_name] = []
        
        # Add general annotation columns
        annotation_columns.update({
            'annotation_sources': [],
            'annotation_success_rate': [],
            'annotation_timestamp': []
        })
        
        # Process each variant
        for _, variant in annotated_df.iterrows():
            variant_id = variant['variant_id']
            results = annotation_results.get(variant_id, [])
            
            # Track which sources provided data
            sources = []
            successful_annotations = 0
            
            # Initialize all annotation fields for this variant
            variant_annotations = {col: None for col in annotation_columns.keys()}
            
            # Process each annotation result
            for result in results:
                if result.success:
                    successful_annotations += 1
                    sources.append(result.source)
                    
                    # Extract specific fields for this API
                    api_fields = settings.annotation_fields.get(result.source, [])
                    for field in api_fields:
                        col_name = f"{result.source}_{field.replace('.', '_')}"
                        value = self._extract_nested_field(result.data, field)
                        variant_annotations[col_name] = value
            
            # Set metadata
            variant_annotations['annotation_sources'] = ','.join(sources)
            variant_annotations['annotation_success_rate'] = (
                successful_annotations / len(results) if results else 0
            )
            variant_annotations['annotation_timestamp'] = datetime.now().isoformat()
            
            # Add to lists
            for col, value in variant_annotations.items():
                annotation_columns[col].append(value)
        
        # Add annotation columns to DataFrame
        for col, values in annotation_columns.items():
            annotated_df[col] = values
        
        return annotated_df
    
    def _extract_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extract a nested field from annotation data using dot notation."""
        try:
            value = data
            for key in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit():
                    idx = int(key)
                    value = value[idx] if idx < len(value) else None
                else:
                    return None
            return value
        except:
            return None
    
    def save_results(self, df: pd.DataFrame, output_path: Path, 
                    formats: List[str] = None) -> Dict[str, Path]:
        """Save results in multiple formats."""
        formats = formats or settings.output_formats
        saved_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    file_path = output_path.with_suffix('.csv')
                    df.to_csv(file_path, index=False)
                    
                elif fmt == 'xlsx':
                    file_path = output_path.with_suffix('.xlsx')
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Annotated_Variants', index=False)
                        
                        # Add summary sheet
                        summary_df = self._create_summary(df)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                elif fmt == 'html':
                    file_path = output_path.with_suffix('.html')
                    self._save_html_report(df, file_path)
                
                elif fmt == 'json':
                    file_path = output_path.with_suffix('.json')
                    df.to_json(file_path, orient='records', indent=2)
                
                saved_files[fmt] = file_path
                logger.info(f"Saved {fmt.upper()} output to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {fmt} format: {e}")
    
    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary DataFrame with key statistics."""
        summary_data = []
        
        # Basic statistics
        summary_data.append({
            'Metric': 'Total Variants',
            'Value': len(df),
            'Description': 'Total number of variants processed'
        })
        
        summary_data.append({
            'Metric': 'Successfully Annotated',
            'Value': (df['annotation_success_rate'] > 0).sum(),
            'Description': 'Variants with at least one successful annotation'
        })
        
        # Quality statistics
        if 'QUAL' in df.columns:
            summary_data.append({
                'Metric': 'Mean Quality Score',
                'Value': f"{df['QUAL'].mean():.2f}",
                'Description': 'Average variant quality score'
            })
        
        return pd.DataFrame(summary_data)
    
    def _save_html_report(self, df: pd.DataFrame, file_path: Path):
        """Save an HTML report with interactive tables."""
        try:
            # Simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Variant Annotation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Variant Annotation Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Total Variants: {len(df)}</p>
                </div>
                
                <div class="section">
                    <h2>Variant Data</h2>
                    {df.head(100).to_html(classes='table', table_id='variants_table')}
                </div>
            </body>
            </html>
            """
            
            with open(file_path, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Error creating HTML report: {e}") 
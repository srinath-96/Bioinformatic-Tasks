"""
Reporter agent for the variant annotation pipeline.
"""

from crewai import Agent
from typing import Dict, Any,List
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from config.agents_config import AgentsConfig
from config.settings import settings
from utils.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class ReporterAgent:
    """Agent responsible for generating comprehensive variant annotation reports."""
    
    def __init__(self):
        self.config = AgentsConfig.REPORTER
        self.data_handler = DataHandler()
        
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
    
    def generate_report(self, annotated_data: pd.DataFrame, 
                       annotation_stats: Dict[str, Any],
                       output_path: Path) -> Dict[str, Any]:
        """Generate a comprehensive variant annotation report."""
        logger.info(f"ReporterAgent: Generating report for {len(annotated_data)} annotated variants")
        
        result = {
            'success': False,
            'output_files': {},
            'report_summary': {},
            'errors': []
        }
        
        try:
            # Prepare the data for reporting
            report_data = self._prepare_report_data(annotated_data, annotation_stats)
            
            # Generate summary statistics
            summary = self._generate_summary(report_data, annotation_stats)
            
            # Save reports in multiple formats
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"variant_annotation_report_{timestamp}"
            report_path = output_path / base_filename
            
            saved_files = self.data_handler.save_results(
                report_data, 
                report_path, 
                formats=settings.output_formats
            )
            
            # Ensure saved_files is a valid dictionary
            if saved_files is None:
                saved_files = {}
            
            # Create summary report
            summary_file = self._create_summary_report(
                summary, annotation_stats, output_path, timestamp
            )
            
            if summary_file:
                saved_files['summary'] = summary_file
            
            # Calculate annotated variants safely
            annotated_count = 0
            if len(report_data) > 0 and 'annotation_success_rate' in report_data.columns:
                annotated_count = (report_data['annotation_success_rate'] > 0).sum()
            
            result.update({
                'success': True,
                'output_files': saved_files,
                'report_summary': summary,
                'total_variants': len(report_data),
                'annotated_variants': annotated_count
            })
            
            logger.info(f"ReporterAgent: Successfully generated {len(saved_files)} report files")
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _prepare_report_data(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """Prepare and enrich data for reporting."""
        report_df = df.copy()
        
        # Add derived columns for better reporting
        if len(report_df) > 0:
            # Variant type classification
            report_df['variant_type'] = report_df.apply(self._classify_variant_type, axis=1)
            
            # Annotation quality score
            report_df['annotation_quality'] = report_df['annotation_success_rate'].apply(
                lambda x: 'High' if x >= 0.8 else 'Medium' if x >= 0.5 else 'Low'
            )
        
        return report_df
    
    def _classify_variant_type(self, row) -> str:
        """Classify variant type based on REF and ALT alleles."""
        try:
            ref = str(row['REF'])
            alt = str(row['ALT']).split(',')[0]  # Take first ALT allele
            
            if len(ref) == 1 and len(alt) == 1:
                return 'SNP'
            elif len(ref) > len(alt):
                return 'Deletion'
            elif len(ref) < len(alt):
                return 'Insertion'
            else:
                return 'Complex'
        except:
            return 'Unknown'
    
    def _generate_summary(self, df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            'overview': {
                'total_variants': len(df),
                'processing_date': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'annotation_success_rate': stats.get('annotation_success_rate', 0)
            },
            'variant_distribution': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        if len(df) == 0:
            return summary
        
        # Variant distribution
        summary['variant_distribution'] = {
            'by_chromosome': df['CHROM'].value_counts().head(10).to_dict(),
            'by_type': df['variant_type'].value_counts().to_dict() if 'variant_type' in df.columns else {}
        }
        
        # Quality metrics
        summary['quality_metrics'] = {
            'mean_quality_score': float(df['QUAL'].mean()) if 'QUAL' in df.columns and df['QUAL'].notna().any() else None,
            'median_annotation_rate': float(df['annotation_success_rate'].median())
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, stats)
        summary['recommendations'] = recommendations
        
        return summary
    
    def _generate_recommendations(self, df: pd.DataFrame, stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        if len(df) == 0:
            recommendations.append("No variants were successfully processed. Check input data quality.")
            return recommendations
        
        # Annotation coverage recommendations
        success_rate = stats.get('annotation_success_rate', 0)
        if success_rate < 0.5:
            recommendations.append("Low annotation success rate. Consider checking API connectivity and variant format.")
        
        if not recommendations:
            recommendations.append("Analysis completed successfully. All metrics are within expected ranges.")
        
        return recommendations
    
    def _create_summary_report(self, summary: Dict[str, Any], stats: Dict[str, Any], 
                             output_path: Path, timestamp: str) -> Path:
        """Create a human-readable summary report."""
        try:
            summary_file = output_path / f"summary_report_{timestamp}.txt"
            
            with open(summary_file, 'w') as f:
                f.write("VARIANT ANNOTATION PIPELINE SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Generated: {summary['overview']['processing_date']}\n")
                f.write(f"Total Variants Processed: {summary['overview']['total_variants']}\n")
                f.write(f"Overall Success Rate: {summary['overview']['annotation_success_rate']:.1%}\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
            
            return summary_file
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            return None 
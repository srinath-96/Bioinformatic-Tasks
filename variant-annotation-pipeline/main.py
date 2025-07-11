"""
Main execution script for the Agentic Variant Annotation Pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic Variant Annotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/input/sample.vcf
  python main.py --input sample.vcf --output /path/to/output/ --formats csv xlsx
  python main.py --input sample.vcf --config custom_config.yaml
  python main.py --input sample.vcf --min-quality 30 --max-variants 1000
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input VCF file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: from config)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file (default: auto-detect config.yaml)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['csv', 'xlsx', 'html', 'json'],
        help='Output formats to generate (default: from config)'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        help='Minimum variant quality score (default: from config)'
    )
    
    parser.add_argument(
        '--max-variants',
        type=int,
        help='Maximum number of variants to process (for testing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def setup_logging(config_file_path: str = None):
    """Set up logging configuration."""
    # Import here to avoid circular imports
    from config.settings import Settings
    
    # Load settings with config file if provided
    if config_file_path:
        settings = Settings.from_yaml(Path(config_file_path))
    else:
        settings = Settings.load()
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.project_root / settings.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), settings

def run_pipeline(input_path: Path, output_path: Path, max_variants: int = None):
    """Run the complete variant annotation pipeline."""
    from agents import DataLoaderAgent, AnnotatorAgent, ReporterAgent
    
    logger = logging.getLogger(__name__)
    
    pipeline_start_time = datetime.now()
    logger.info("🧬 Starting Agentic Variant Annotation Pipeline")
    logger.info("=" * 60)
    
    # Initialize agents
    logger.info("Initializing agents...")
    data_loader = DataLoaderAgent()
    annotator = AnnotatorAgent()
    reporter = ReporterAgent()
    
    pipeline_results = {
        'success': False,
        'stages': {},
        'output_files': {},
        'errors': [],
        'stats': {}
    }
    
    try:
        # Stage 1: Data Loading and Validation
        logger.info("🔍 STAGE 1: Data Loading and Validation")
        logger.info("-" * 40)
        
        data_result = data_loader.load_and_validate_vcf(input_path)
        pipeline_results['stages']['data_loading'] = data_result
        
        if not data_result['success']:
            logger.error("Data loading failed!")
            pipeline_results['errors'].extend(data_result['errors'])
            return pipeline_results
        
        variants_df = data_result['data']
        logger.info(f"✅ Loaded {len(variants_df)} variants")
        
        # Apply max variants limit if specified
        if max_variants and len(variants_df) > max_variants:
            variants_df = variants_df.head(max_variants)
            logger.info(f"Limited to first {max_variants} variants for processing")
        
        # Stage 2: Variant Annotation
        logger.info("🔬 STAGE 2: Variant Annotation")
        logger.info("-" * 40)
        
        annotation_result = annotator.annotate_variants(variants_df)
        pipeline_results['stages']['annotation'] = annotation_result
        
        if not annotation_result['success']:
            logger.error("Annotation failed!")
            pipeline_results['errors'].extend(annotation_result['errors'])
            return pipeline_results
        
        annotated_df = annotation_result['annotated_data']
        annotation_stats = annotation_result['stats']
        logger.info(f"✅ Annotated {len(annotated_df)} variants")
        
        # Stage 3: Report Generation
        logger.info("📊 STAGE 3: Report Generation")
        logger.info("-" * 40)
        
        report_result = reporter.generate_report(
            annotated_df, 
            annotation_stats, 
            output_path
        )
        pipeline_results['stages']['reporting'] = report_result
        
        if not report_result['success']:
            logger.error("Report generation failed!")
            pipeline_results['errors'].extend(report_result['errors'])
            return pipeline_results
        
        logger.info(f"✅ Generated {len(report_result['output_files'])} report files")
        
        # Pipeline Success
        pipeline_end_time = datetime.now()
        total_time = (pipeline_end_time - pipeline_start_time).total_seconds()
        
        pipeline_results.update({
            'success': True,
            'output_files': report_result['output_files'],
            'stats': {
                'total_processing_time': total_time,
                'variants_processed': len(annotated_df),
                'annotation_stats': annotation_stats,
                'data_loading_stats': data_result['stats']
            }
        })
        
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Variants Processed: {len(annotated_df)}")
        logger.info(f"Output Files: {len(report_result['output_files'])}")
        logger.info("Output files:")
        for format_type, file_path in report_result['output_files'].items():
            logger.info(f"  {format_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        pipeline_results['errors'].append(f"Pipeline error: {str(e)}")
    
    return pipeline_results

def main():
    """Main entry point."""
    # Parse arguments first to get config file path
    args = parse_arguments()
    
    # Setup logging with potential config file
    logger, settings = setup_logging(args.config)
    
    try:
        # Validate inputs
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = settings.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Update settings based on arguments
        if args.min_quality:
            settings.min_quality_score = args.min_quality
        
        if args.formats:
            settings.output_formats = args.formats
        
        if args.verbose:
            settings.log_level = "DEBUG"
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Display configuration info
        logger.info("🔧 Configuration:")
        logger.info(f"  LLM Provider: {settings.llm_config.provider}")
        logger.info(f"  LLM Model: {settings.llm_config.model}")
        logger.info(f"  Output Path: {output_path}")
        logger.info(f"  Output Formats: {settings.output_formats}")
        logger.info(f"  Min Quality: {settings.min_quality_score}")
        if args.max_variants:
            logger.info(f"  Max Variants: {args.max_variants}")
        
        # Run pipeline
        results = run_pipeline(input_path, output_path, args.max_variants)
        
        if results['success']:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            for error in results['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 
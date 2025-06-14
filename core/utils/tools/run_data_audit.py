from pathlib import Path
from data_auditor import DataAuditor
from data_cleaner import DataCleaner
from data_logger import DataLogger
import logging

logger = logging.getLogger(__name__)

def main():
    # Setup paths
    data_root = Path(__file__).parent.parent / "data"
    
    # Run audit
    auditor = DataAuditor(data_root)
    inventory = auditor.scan_files()
    
    #data logger 
    DataLogger.log_dir = data_root / "logs"
    data_logger = DataLogger(data_root / "logs")
    data_logger.log_event("audit_start", "Starting data audit", {"data_root": str(data_root)})
    data_logger.log_info("Inventory scanned", {"inventory": inventory})
    data_logger.log_event("audit_end", "Data audit completed", {"inventory": inventory})
    # Validate files
    for ext, files in inventory.items():
        for file_path in files:
            validation_result = auditor.validate_file(file_path)
            if not validation_result["valid"]:
                data_logger.log_error(f"Validation failed for {file_path}", validation_result)
            else:
                data_logger.log_info(f"Validation successful for {file_path}", validation_result)
    # Clean data
    cleaner = DataCleaner(data_root)
    removed = cleaner.clean_duplicates(auditor.file_hashes)
    
    # Generate report
    report = auditor.generate_report()
    report_path = data_root / "data_audit_report.md"
    report_path.write_text(report)
    
if __name__ == "__main__":
    main()

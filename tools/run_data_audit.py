from pathlib import Path
from data_auditor import DataAuditor
from data_cleaner import DataCleaner
import logging

logger = logging.getLogger(__name__)

def main():
    # Setup paths
    data_root = Path(__file__).parent.parent / "data"
    
    # Run audit
    auditor = DataAuditor(data_root)
    inventory = auditor.scan_files()
    
    # Clean data
    cleaner = DataCleaner(data_root)
    removed = cleaner.clean_duplicates(auditor.file_hashes)
    
    # Generate report
    report = auditor.generate_report()
    report_path = data_root / "data_audit_report.md"
    report_path.write_text(report)
    
if __name__ == "__main__":
    main()

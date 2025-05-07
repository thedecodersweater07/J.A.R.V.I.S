import pandas as pd
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class VirtualEnvSummarizer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.reports_path = self.data_path / "reports"
        self.reports_path.mkdir(exist_ok=True)

    def generate_summary(self) -> str:
        """Generate CSV summary from virtual environment data"""
        # Collect all step data
        step_data = []
        for file in self.data_path.glob("step_data_*.jsonl.gz"):
            with gzip.open(file, 'rt') as f:
                for line in f:
                    data = json.loads(line)
                    step_data.append(self._process_step_data(data))

        if not step_data:
            return ""

        # Create DataFrame and save CSV
        df = pd.DataFrame(step_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.reports_path / f"virtual_env_summary_{timestamp}.csv"
        
        # Add summary statistics
        summary_stats = self._calculate_summary_stats(df)
        df = pd.concat([df, summary_stats], axis=0)
        
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def _process_step_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from step data"""
        return {
            'timestamp': data['timestamp'],
            'action_type': data['action'].get('type', 'unknown'),
            'process_time': data['result'].get('process_time', 0),
            'coherence': data['result'].get('coherence', 0),
            'state_complexity': len(str(data['state']))
        }

    def _calculate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics for numeric columns"""
        stats = df.describe().loc[['mean', 'min', 'max']]
        stats.index = [f'SUMMARY_{i.upper()}' for i in stats.index]
        return stats

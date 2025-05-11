import spacy
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
import subprocess
from pathlib import Path
from typing import List, Dict

class SystemInstaller:
    def __init__(self):
        self.console = Console()
        self.required_models = [
            "nl_core_news_sm",
            "nl_core_news_lg",
            "en_core_web_sm"
        ]
        self.required_packages = [
            "torch",
            "transformers",
            "spacy",
            "rich"
        ]

    def install(self):
        """Run full system installation"""
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Install packages
            task1 = progress.add_task("[green]Installing packages...", total=len(self.required_packages))
            self._install_packages(progress, task1)

            # Install models
            task2 = progress.add_task("[blue]Installing language models...", total=len(self.required_models))
            self._install_models(progress, task2)

            # Configure system
            task3 = progress.add_task("[yellow]Configuring system...", total=100)
            self._configure_system(progress, task3)

    def _install_packages(self, progress, task_id):
        """Install required Python packages"""
        for package in self.required_packages:
            try:
                subprocess.check_call(["pip", "install", "-q", package])
                progress.advance(task_id)
            except Exception as e:
                self.console.print(f"[red]Failed to install {package}: {e}")

    def _install_models(self, progress, task_id):
        """Install required language models"""
        for model in self.required_models:
            try:
                spacy.cli.download(model)
                progress.advance(task_id)
            except Exception as e:
                self.console.print(f"[red]Failed to install model {model}: {e}")

    def _configure_system(self, progress, task_id):
        """Configure system settings"""
        steps = [20, 40, 60, 80, 100]
        for step in steps:
            # Add configuration steps here
            progress.update(task_id, completed=step)

if __name__ == "__main__":
    installer = SystemInstaller()
    installer.install()

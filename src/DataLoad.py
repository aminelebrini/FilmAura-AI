from pathlib import Path
import os
from json import dump
from rich.console import Console
class DataLoad:
    def __init__(self):
        pass

    def load_data(self, data_json, file_path):
        console = Console()

        if file_path.is_dir():
            raise ValueError(
                f"Expected a file path, but found a directory: {file_path}"
            )

        parent_dir = file_path.parent

        if not parent_dir.exists():
            print("File does not exist! Starting raw extraction...")
            with console.status("Creating bronze data file...", spinner="dots"):
                os.makedirs(parent_dir, exist_ok=True)

            print(f"Bronze data file created at: {file_path}")

        with open(file_path, "w") as f:
            dump(data_json, f, ensure_ascii=False, indent=4)
            console.print(
                f"[bold green]✓ Data loaded successfully to:[bold green] {file_path}"
            )

            
            



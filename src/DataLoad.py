from pathlib import Path
import os
from json import dump
from rich.console import Console
class DataLoad:
    def __init__(self):
        
        BASE_DIR = Path(__file__).parent.parent
        self.bronze_path = BASE_DIR / "data" / "bronze" / "data_bronze.json"

    def load_silver_data(self, data_json):
        console = Console()
        if not self.bronze_path.exists():
            print("File does not exist! Starting raw extraction...")

            with console.status("Creating bronze data file...", spinner="dots"):
                os.makedirs(self.bronze_path.parent, exist_ok=True)

            print(f"Bronze data file created at: {self.bronze_path}")

        with open(self.bronze_path, "w") as f:
            dump(data_json, f, indent=4)
            print(f"Data loaded to bronze data file at: {self.bronze_path}")
            
            



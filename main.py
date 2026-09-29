from src.DataExtractor import DataExtractor
from src.DataLoad import DataLoad
from src.DataCleaning import DataCleaning
from pathlib import Path
from json import load
def main():
    data_load = DataLoad()
    data_extractor = DataExtractor()
    data_cleaning = DataCleaning()


    BASE_DIR = Path(__file__).parent
    bronze_path = BASE_DIR / "data" / "bronze" / "data_bronze.json"

    # result = data_extractor.extract_all_data(100)
    # data_load.load_silver_data(result)
    with open(bronze_path, "r") as f:
            films_data = load(f)
    
    cleaned_data = data_cleaning.clean_data(films_data)
    print(cleaned_data)

if __name__ == "__main__":
    main()
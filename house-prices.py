import os
import kaggle
import pandas as pd
from pathlib import Path
import zipfile


def download_competition_data(competition: str, data_dir: Path) -> Path:
    kaggle.api.competition_download_files(competition, path=data_dir)
    zip_path = data_dir / f"{competition}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Error: {zip_path} not found")
    return zip_path


def process_csv_files(data_dir: Path) -> None:
    # List all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    print("\nAvailable CSV files:")
    for file in csv_files:
        print(f"- {file.name}")

    # Read and display the first 5 rows of each CSV file
    print("\nFirst 5 rows of each file:")
    for file in csv_files:
        print(f"\n{file.name}:")
        df = pd.read_csv(file)
        print(df.head())


def setup_kaggle():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Download competition data
    competition = "home-data-for-ml-course"
    zip_path = download_competition_data(competition, data_dir)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
        print(f"Data extracted successfully to {data_dir}")
        process_csv_files(data_dir)


if __name__ == "__main__":
    setup_kaggle()

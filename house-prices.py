import kaggle
import pandas as pd
from pathlib import Path
from zipfile import ZipFile


# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download competition data
competition = "home-data-for-ml-course"
kaggle.api.competition_download_files(competition, path=data_dir)
zip_path = data_dir / f"{competition}.zip"
assert zip_path.exists()

# Extract the zip file
with ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

csv_files = list(data_dir.glob("*.csv"))

# Read and display the first 5 rows of each CSV file
for file in csv_files:
    df = pd.read_csv(file)
    print(df.head())

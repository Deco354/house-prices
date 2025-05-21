import kaggle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

# View Columns and select features and target variable

train_df.columns
features = ["GrLivArea", "YearBuilt", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = train_df[features]
train_df.GrLivArea.mean()
train_df.GrLivArea.std()
y = train_df.SalePrice
x.describe()

# Check if data has na values
x.isna().sum()

# Create histogram of GrLivArea using pandas to check if there are significant outliers
train_df["GrLivArea"].hist()
y.hist()

# Scale the features and target variable
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = (y - y.mean()) / y.std()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    x_scaled, y_scaled, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

# Calculate metrics
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Validation R²: {val_r2:.2f}")

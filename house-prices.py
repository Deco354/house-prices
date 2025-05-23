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

data_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

# View Columns and select features and target variable

data_df.columns
features = ["GrLivArea", "YearBuilt", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = data_df[features]
data_df.GrLivArea.mean()
data_df.GrLivArea.std()
y = data_df.SalePrice
x.describe()

# Check if data has na values
x.isna().sum()

# Check for outliers
data_df["GrLivArea"].hist()
data_df["SalePrice"].hist()
data_df["GrLivArea"].describe()
data_df["SalePrice"].describe()

# Split the data into training and validation sets
# It's important to split the data before scaling the features
# because the scaler will use the training data to scale the features
# and the validation data will be scaled using the training data statistics
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features and target variable
# We use fit_transform on the training data to learn the mean and standard deviation of the training data
# We use transform on the validation data to scale it using the training data statistics.
# The mean and standard deviation of the validation data should not be used because it will bias the model
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# Create and train the model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Make predictions
train_predictions = model.predict(x_train_scaled)
val_predictions = model.predict(x_val_scaled)

# Calculate metrics
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Validation R²: {val_r2:.2f}")

# Make predictions on the test set
## See how sample submission is structured
print(sample_submission_df)

## Make predictions on the test set
x_test = test_df[features]
x_test_scaled = scaler.transform(x_test)
test_predictions = model.predict(x_test_scaled)

# Transform predictions back to original scale
test_predictions_original = test_predictions * y.std() + y.mean()

# Create submission file
submission_df = pd.DataFrame(
    {"Id": test_df.Id, "SalePrice": test_predictions_original.flatten()}
)
submission_csv = submission_df.to_csv("submission.csv", index=False)

# Upload to Kaggle
if False:
    kaggle.api.competition_submit(
        "submission.csv",
        "House Price Prediction",
        competition,
    )

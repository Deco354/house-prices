import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create data directory if it doesn't exist
iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
if iskaggle:
    data_dir = Path("/kaggle/input/home-data-for-ml-course")
else:
    import kaggle

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

data_df_original = pd.read_csv(data_dir / "train.csv", index_col="Id")
test_df_original = pd.read_csv(data_dir / "test.csv", index_col="Id")
data_df = data_df_original.copy()
test_df = test_df_original.copy()
sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

# Split the data into training and validation sets before any preprocessing
x = data_df.drop("SalePrice", axis=1)
y = data_df["SalePrice"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Select Features
## Make a list of numeric features
numeric_features = data_df.select_dtypes(include="number").columns
numeric_features = numeric_features.drop("SalePrice")

## Analyze NA values
na_feature_counts = data_df[numeric_features].isna().sum()
na_feature_counts = na_feature_counts[na_feature_counts > 0]

## Drop features with too many NA values
features_to_drop = ["LotFrontage", "GarageYrBlt"]  # Based on initial analysis
numeric_features = numeric_features.drop(features_to_drop)


# Preprocess data function, we'll need to keep this step consistent for all our datasets
def preprocess_data(df, selected_columns):
    # Select only the chosen columns
    df = df[selected_columns]
    # Fill NA values with 0
    df = df.fillna(0)
    return df


# Preprocess training and validation sets
x_train_processed = preprocess_data(x_train, numeric_features)
x_val_processed = preprocess_data(x_val, numeric_features)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_processed)
x_val_scaled = scaler.transform(x_val_processed)

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

# Preprocess test set using the same selected columns
test_processed = preprocess_data(test_df, numeric_features)
test_scaled = scaler.transform(test_processed)
test_predictions = model.predict(test_scaled)

# Create submission file
submission_df = pd.DataFrame(
    {"Id": test_df.index, "SalePrice": test_predictions.flatten()}
)
submission_csv = submission_df.to_csv("submission.csv", index=False)

# Upload to Kaggle
if False and not iskaggle:
    kaggle.api.competition_submit(
        "submission.csv",
        "House Price Prediction",
        competition,
    )

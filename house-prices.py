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

data_df_original = pd.read_csv(data_dir / "train.csv")
test_df_original = pd.read_csv(data_dir / "test.csv")
data_df = data_df_original.copy()
test_df = test_df_original.copy()
sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

# Create feature list only of numeric columns minus the Target variable
data_df.columns
features = data_df.select_dtypes(include="number").columns
features = features.drop("SalePrice")

# Check for na values in all datasets
all_data_df = pd.concat([data_df_original, test_df_original])
na_feature_counts = all_data_df[features].isna().sum()
na_feature_counts = na_feature_counts[na_feature_counts > 0]
## LotFrontage and GarageYrBlt have a lot of na values so we'll drop them for now
features = features.drop(["LotFrontage", "GarageYrBlt"])


## Fill na values with 0
def handle_na(df):
    return df.fillna(0)


## Preprocess data
def preprocess_data(df, columns):
    df = df[columns]
    df = handle_na(df)
    return df


selected_columns = list(features) + ["SalePrice"]
data_df = preprocess_data(data_df, selected_columns)

# Split the data into training and validation sets and check their distributions
# It's important to split the data before scaling the features
# because the scaler will use the training data to scale the features
# and the validation data will be scaled using the training data statistics
x = data_df.drop("SalePrice", axis=1)
y = data_df["SalePrice"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
pd.concat([x_train, y_train], axis=1).describe()
pd.concat([x_val, y_val], axis=1).describe()

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

## This initial model is not very accurate so the validation R2 will vary quite a bit
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Validation R²: {val_r2:.2f}")

# Make predictions on the test set
## See how sample submission is structured
print(sample_submission_df)

## Check test set n/as
test_df_original
test_df.isna().sum()

## Preprocess test set
test_df = preprocess_data(test_df, features)
test_scaled = scaler.transform(test_df)
test_predictions = model.predict(test_scaled)

# Create submission file
submission_df = pd.DataFrame(
    {"Id": test_df.Id, "SalePrice": test_predictions.flatten()}
)
submission_csv = submission_df.to_csv("submission.csv", index=False)

# Upload to Kaggle
if False and not iskaggle:
    kaggle.api.competition_submit(
        "submission.csv",
        "House Price Prediction",
        competition,
    )

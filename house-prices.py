import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

SHOULD_SUBMIT_TO_KAGGLE = False
RANDOM_SEED = 42

# Create data directory if it doesn't exist
is_running_on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
if is_running_on_kaggle:
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

data_df = pd.read_csv(data_dir / "train.csv", index_col="Id")
test_df = pd.read_csv(data_dir / "test.csv", index_col="Id")
sample_submission_df = pd.read_csv(data_dir / "sample_submission.csv")

# Split the data into training and validation sets before any preprocessing
x = data_df.drop("SalePrice", axis=1)
y = data_df["SalePrice"]
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_SEED
)

# Select Features
## Make a list of numeric features
numeric_features = data_df.select_dtypes(include="number").columns
numeric_features = numeric_features.drop("SalePrice")


# Analyze NA values
def list_na_features(df):
    na_feature_counts = df[numeric_features].isna().sum()
    return na_feature_counts[na_feature_counts > 0]


## Check there isn't a large discrepency in na values between training and test sets (there isn't)
list_na_features(data_df)
list_na_features(test_df)

## Take a look at features with significant na values
### LotFrontage: Linear feet of street connected to property
### This is a measure of the street connected to the property
### It's possible that 0 values mean there is no street connected to the property or it could just be a missing value
#### Having 0 lot frontage would likely be a negative factor in the sale price
#### Let's compare the median sale price of properties with 0 lot frontage to the median sale price of properties with non-0 lot frontage

not_na_lot_frontage_df = data_df[~data_df.LotFrontage.isna()]
data_df.SalePrice.median()
not_na_lot_frontage_df.SalePrice.median()

#### The median sale prices aren't that different, meaning the na values are missing recordings rather than 0

#### Apartments will often have 0 lot frontage, let's check if any exist within our dataset
data_df.BldgType.unique()
#### None of these are apartments, I think it's safe to assume these values are missing and not 0


## na_feature descriptions:
## GarageYrBlt: Year garage was built
## MasVnrArea: Masonry veneer area in square feet


## Drop features with too many NA values
features_to_drop = ["LotFrontage", "GarageYrBlt"]  # Based on initial analysis
numeric_features = numeric_features.drop(features_to_drop)


# Scale the features, only fit the scaler on training data
def scale_data(df, scaler: StandardScaler, is_training: bool):
    if is_training:
        return scaler.fit_transform(df)
    else:
        return scaler.transform(df)


# Preprocess data function, we'll need to keep this step consistent for all our datasets
def preprocess_data(
    df, feature_column_names, scaler: StandardScaler, is_training: bool
):
    df = df[feature_column_names]
    df = df.fillna(0)
    df = scale_data(df, scaler, is_training)
    return df


# Preprocess training and validation sets
scaler = StandardScaler()
x_train_processed = preprocess_data(
    x_train,
    numeric_features,
    scaler,
    is_training=True,
)
x_val_processed = preprocess_data(
    x_val,
    numeric_features,
    scaler,
    is_training=False,
)

# Create and train the model
model = LinearRegression()
model.fit(x_train_processed, y_train)

# Make predictions
train_predictions = model.predict(x_train_processed)
val_predictions = model.predict(x_val_processed)

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
test_processed = preprocess_data(
    test_df,
    numeric_features,
    scaler,
    is_training=False,
)
test_predictions = model.predict(test_processed)

# Create submission file
submission_df = pd.DataFrame(
    {"Id": test_df.index, "SalePrice": test_predictions.flatten()}
)
submission_csv = submission_df.to_csv("submission.csv", index=False)

# Upload to Kaggle
if SHOULD_SUBMIT_TO_KAGGLE and not is_running_on_kaggle:
    kaggle.api.competition_submit(
        "submission.csv",
        "House Price Prediction",
        competition,
    )

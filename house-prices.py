import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import os

SHOULD_SUBMIT_TO_KAGGLE = True
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

## Make a list of category features
categorical_features = data_df.select_dtypes(include="object").columns

# Create pipeline
## Our preprocessing has grown sufficiently complex to justify using a pipeline
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "one_hot",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
        ),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = LinearRegression()
# model = RandomForestRegressor(n_estimators=100, random_state=0)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
pipeline.fit(x_train, y_train)

# Make predictions
train_predictions = pipeline.predict(x_train)
val_predictions = pipeline.predict(x_val)

# Calculate metrics
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)
train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Validation R²: {val_r2:.2f}")

test_predictions = pipeline.predict(test_df)

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

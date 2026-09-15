import pandas as pd
import numpy as np
import re


# ============================================================
# 1. FILE PATHS
# ============================================================

input_path = r"D:\projects\parkinson_detection_project\raw_dataset\tremors\dataset.csv"
output_path = r"D:\projects\parkinson_detection_project\cleaned_dataset\tremors_clean.csv"


# ============================================================
# 2. LOAD RAW CSV
# ============================================================

# Load everything as text initially.
# This prevents pandas from changing values such as "-"
# during the initial loading process.

raw = pd.read_csv(
    input_path,
    encoding="latin1",
    header=None,
    dtype=str,
    keep_default_na=False
)

print("=" * 65)
print("RAW DATASET")
print("=" * 65)

print("Original shape:", raw.shape)


# ============================================================
# 3. UNDERSTAND THE TWO HEADER ROWS
# ============================================================

# Row 0 = group/category information
# Row 1 = actual variable names
# Row 2 onwards = participant data

group_headers = raw.iloc[0].copy()
variable_headers = raw.iloc[1].copy()

data = raw.iloc[2:].copy().reset_index(drop=True)


# ============================================================
# 4. CREATE CLEAN COLUMN NAMES
# ============================================================

def clean_name(value):

    value = str(value).strip()

    # Replace line breaks
    value = value.replace("\n", " ")

    # Lowercase
    value = value.lower()

    # Replace special characters with underscore
    value = re.sub(r"[^a-z0-9]+", "_", value)

    # Remove unnecessary underscores
    value = value.strip("_")

    return value


new_columns = []

for i in range(len(variable_headers)):

    variable = clean_name(variable_headers.iloc[i])
    group = clean_name(group_headers.iloc[i])

    # The first column has no group name
    if group and group != "nan":

        column_name = f"{group}_{variable}"

    else:

        column_name = variable

    new_columns.append(column_name)


# ============================================================
# 5. MAKE COLUMN NAMES UNIQUE
# ============================================================

column_counter = {}
unique_columns = []

for column in new_columns:

    if column not in column_counter:

        column_counter[column] = 1
        unique_columns.append(column)

    else:

        column_counter[column] += 1

        unique_columns.append(
            f"{column}_{column_counter[column]}"
        )


data.columns = unique_columns


# ============================================================
# 6. CLEAN CELL WHITESPACE
# ============================================================

for column in data.columns:

    data[column] = data[column].apply(
        lambda x:
            x.strip()
            if isinstance(x, str)
            else x
    )


# ============================================================
# 7. STANDARDIZE ONLY TRUE EMPTY VALUES
# ============================================================

# IMPORTANT:
# "-" is NOT treated as missing.
# It is meaningful information in this dataset.

true_missing_values = [
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NULL",
    "null",
    "?"
]

data = data.replace(
    true_missing_values,
    np.nan
)


# ============================================================
# 8. STANDARDIZE YES / NO
# ============================================================

for column in data.columns:

    if data[column].dtype == "object":

        data[column] = data[column].replace({

            "yes": "Yes",
            "YES": "Yes",

            "no": "No",
            "NO": "No"
        })


# ============================================================
# 9. STANDARDIZE GENDER
# ============================================================

for column in data.columns:

    if data[column].dtype == "object":

        data[column] = data[column].replace({

            "female": "F",
            "Female": "F",

            "male": "M",
            "Male": "M"
        })


# ============================================================
# 10. CONVERT ONLY FULLY NUMERIC COLUMNS
# ============================================================

# We convert a column to numeric ONLY if every non-empty
# value in that column is numeric.
#
# This prevents "-" from becoming NaN.

for column in data.columns:

    if data[column].dtype != "object":
        continue

    non_empty = data[column].dropna()

    if len(non_empty) == 0:
        continue

    converted = pd.to_numeric(
        non_empty,
        errors="coerce"
    )

    # Convert only if ALL non-empty values are numeric
    if converted.notna().all():

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


# ============================================================
# 11. CHECK THAT WE DID NOT CREATE UNEXPECTED MISSING VALUES
# ============================================================

missing_report = data.isna().sum()

total_missing = missing_report.sum()

print("\nTotal missing values after cleaning:",
      total_missing)

if total_missing > 0:

    print("\nColumns containing missing values:")

    print(
        missing_report[
            missing_report > 0
        ]
    )

else:

    print("No missing values found.")


# ============================================================
# 12. CHECK DUPLICATE ROWS
# ============================================================

duplicate_rows = data.duplicated().sum()

print("\nDuplicate rows:", duplicate_rows)

# Do NOT delete duplicate rows automatically.


# ============================================================
# 13. CHECK PARTICIPANT IDs
# ============================================================

participant_columns = [
    column
    for column in data.columns
    if "participant_code" in column
]


if participant_columns:

    participant_column = participant_columns[0]

    print(
        "\nParticipant ID column:",
        participant_column
    )

    duplicate_ids = data[
        participant_column
    ].duplicated().sum()

    print(
        "Duplicate participant IDs:",
        duplicate_ids
    )


# ============================================================
# 14. CHECK INFINITE VALUES
# ============================================================

numeric_columns = data.select_dtypes(
    include=np.number
).columns


if len(numeric_columns) > 0:

    infinite_count = np.isinf(
        data[numeric_columns].to_numpy()
    ).sum()

    print(
        "\nInfinite values:",
        infinite_count
    )

else:

    print("\nNo numeric columns found.")


# ============================================================
# 15. DATA TYPE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DATA TYPES")
print("=" * 65)

print(data.dtypes)


# ============================================================
# 16. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL DATASET")
print("=" * 65)

print("Rows:", data.shape[0])
print("Columns:", data.shape[1])


# ============================================================
# 17. SAVE CLEANED DATASET
# ============================================================

data.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("CLEANING COMPLETED")
print("=" * 65)

print("\nSaved to:")
print(output_path)

print("\nFinal shape:")
print(data.shape)
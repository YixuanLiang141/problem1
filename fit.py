import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def classify_variables_for_scaling(df, numerical_vars):
    """
    Specify standard_vars and log_vars by skewness
    """

    log_vars = []
    standard_vars = []

    for var in numerical_vars:
        # features with skewness > 1 become log vars, else standard vars
        if var == 'gender':
            continue
        col_data = df[var].dropna()
        if col_data.min() > 0:
            skewness = col_data.skew()
            if abs(skewness) > 1:
                log_vars.append(var)
            else:
                standard_vars.append(var)
        else:
            standard_vars.append(var)

    return standard_vars, log_vars

def preprocess_train_test_with_knn_and_scaling(
    train_csv: str,
    test_csv: str,
    label_col: str,
    n_neighbors: int = 13,
    output_train_csv: str = None,
    output_test_csv: str = None
):
    # ======================
    # 1. Read data
    # ======================
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Drop unused columns
    train_df.drop(['subject_id','stay_id','suspected_infection_time','sofa_time','intime'], axis=1, inplace=True)
    test_df.drop(['subject_id','stay_id','suspected_infection_time','sofa_time','intime'], axis=1, inplace=True)

    # May drop or keep these statistical features
    #train_df.drop([col for col in train_df.columns if '_freq' in col], axis=1, inplace=True)
    train_df.drop([col for col in train_df.columns if '_lasttime' in col], axis=1, inplace=True)
    train_df.drop([col for col in train_df.columns if '_sampled' in col], axis=1, inplace=True)
    train_df.drop([col for col in train_df.columns if '_std' in col], axis=1, inplace=True)
    #test_df.drop([col for col in test_df.columns if '_freq' in col], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if '_lasttime' in col], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if '_sampled' in col], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if '_std' in col], axis=1, inplace=True)

    # Split labels
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col])
    print(X_train.columns)

    if 'gender' in X_train.columns:
        mapping = {'M': 1, 'F': 0}
        X_train['gender'] = X_train['gender'].map(mapping)
        X_test['gender'] = X_test['gender'].map(mapping)

    # ======================
    # 2. Specify scaling variables
    # ======================

    scaling_cols = [
        col for col in X_train.columns
        if (
            col not in ["gender"]
            and "_sampled" not in col
        )
    ]

    standard_vars, log_vars = classify_variables_for_scaling(
        X_train, scaling_cols
    )

    # ======================
    # 3. KNN Imputation
    # ======================
    
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Excluded columns
    base_exclude = {"gender", "age"}

    # Columns to impute
    impute_cols = [
        col for col in X_train.columns
        if (
            col not in base_exclude
            and "_freq" not in col
            and "_sampled" not in col
        )
    ]

    # Copy dataframes
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    # Only fit + transform on specified columns
    X_train_imputed[impute_cols] = imputer.fit_transform(
        X_train[impute_cols]
    )

    # Fit on test set
    X_test_imputed[impute_cols] = imputer.transform(
        X_test[impute_cols]
    )

    # ======================
    # 4. Log transform
    # ======================
    for col in log_vars:
        X_train_imputed[col] = np.log1p(X_train_imputed[col])
        X_test_imputed[col] = np.log1p(X_test_imputed[col])

    # ======================
    # 5. Standard scaling (fit with train set only)
    # ======================
    scaler = StandardScaler()

    scale_vars = standard_vars + log_vars

    X_train_imputed[scale_vars] = scaler.fit_transform(
        X_train_imputed[scale_vars]
    )

    X_test_imputed[scale_vars] = scaler.transform(
        X_test_imputed[scale_vars]
    )

    # ======================
    # 6. Put back labels
    # ======================
    train_processed = X_train_imputed.copy()
    train_processed[label_col] = y_train.values

    test_processed = X_test_imputed.copy()
    test_processed[label_col] = y_test.values

    # ======================
    # 7. Save
    # ======================
    if output_train_csv is None:
        output_train_csv = train_csv.replace(".csv", "_processed.csv")

    if output_test_csv is None:
        output_test_csv = test_csv.replace(".csv", "_processed.csv")

    train_processed.to_csv(output_train_csv, index=False)
    test_processed.to_csv(output_test_csv, index=False)

    print("Preprocessing complete:")
    print(f"  Train → {output_train_csv}  shape={train_processed.shape}")
    print(f"  Test  → {output_test_csv}  shape={test_processed.shape}")

    return train_processed, test_processed

def preprocess_all_folds(csv_prefix, split = 5, label_col = "sepsis"):

    i = 1
    while i <= split:
        preprocess_train_test_with_knn_and_scaling(
            train_csv=csv_prefix + "_train_" + str(i) + ".csv",
            test_csv=csv_prefix + "_test_" + str(i) + ".csv",
            label_col=label_col,
            n_neighbors=13
        )
        i += 1

if __name__ == "__main__":
    preprocess_all_folds("../first/dataset_first")
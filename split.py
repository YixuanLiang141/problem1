import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_and_save_stratified_folds(
    csv_path: str,
    label_col: str,
    n_outer_folds: int = 5,
    random_state: int = 42
):
    """
    Split dataset into n outer folds for nested CV (5 by default).

    Parameters:
    csv_path (str)         : input dataset file
    label_col (str)        : prediction label column, sepsis by default
    n_outer_folds (int)    : number of outer folds, 5 by default
    random_state (int)     : random state, 42 by default
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    skf = StratifiedKFold(
        n_splits=n_outer_folds,
        shuffle=True,
        random_state=random_state
    )

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_path = os.path.join(
            out_dir, f"{base_name}_train_{fold_idx}.csv"
        )
        test_path = os.path.join(
            out_dir, f"{base_name}_test_{fold_idx}.csv"
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(
            f"Fold {fold_idx}: "
            f"train={train_df.shape}, test={test_df.shape}"
        )
if __name__ == "__main__":
    split_and_save_stratified_folds(
        csv_path="../first/dataset_first.csv",
        label_col="sepsis",
        n_outer_folds=5
    )
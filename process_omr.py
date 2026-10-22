import pandas as pd

def process_omr_weight(input_csv, output_csv):
    """
    Read omr.csv in MIMIC-IV and：
    1. Keep result_name == "Weight (Lbs)"
    2. Keep result_value between 50~600
    3. For each subject_id：
        - Keep the earliest chartdate
        - If there is a draw，keep the minimum seq_num
    4. Only keep two columns：
        - subject_id
        - weight
    5. Export to omr_weight.csv
    """

    df = pd.read_csv(input_csv)

    df["result_value"] = pd.to_numeric(df["result_value"], errors="coerce")
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")

    df["chartdate"] = pd.to_datetime(df["chartdate"], errors="coerce")

    df = df[
        (df["result_name"] == "Weight (Lbs)") &
        (df["result_value"] >= 50) &
        (df["result_value"] <= 600)
    ]

    df = df.sort_values(
        by=["subject_id", "chartdate", "seq_num"],
        ascending=[True, True, True]
    )

    df = df.drop_duplicates(
        subset="subject_id",
        keep="first"
    )

    df = df[["subject_id", "result_value"]]

    df = df.rename(columns={
        "result_value": "weight"
    })

    df.to_csv(output_csv, index=False)

    print(f"Done, {len(df)} subject_ids were kept")

if __name__ == "__main__":
    process_omr_weight(
        "../../../hosp/omr.csv",
        "../omr_weight.csv"
    )
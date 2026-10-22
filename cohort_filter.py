import pandas as pd

def cohort_filter(sepsis_file, other_file, ver, min_hours, max_hours, missing_threshold, output_file):
    """
    Filter cohorts based on their SOFA time and number of missing columns.
    Only the ICU admission with the least missing columns are kept for one patient.

    Parameters:
    sepsis_file (str)         : input CSV file of sepsis cohorts
    other_file (str)          : input CSV file of control cohorts
    ver (int)                 : MIMIC version, 3 or 4
    min_hours (float)         : earliest SOFA time compared to ICU admission time in hours, 0 by default
    max_hours (float)         : latest SOFA time compared to ICU admission time in hours, 6 by default
    missing_threshold (float) : rows with missing columns more than this threshold will be removed, 6.8 by default, decimals recommended
    output_file (str)         : output CSV file
    """
    if ver != 3 and ver != 4:
        raise ValueError("Bad version of MIMIC.")
    
    s_df = pd.read_csv(sepsis_file)
    o_df = pd.read_csv(other_file)

    # Filter with sofa time
    if ver == 4:
        s_df['intime'] = pd.to_datetime(s_df['intime'], errors='coerce')
        s_df['sofa_time'] = pd.to_datetime(s_df['sofa_time'], errors='coerce')
        time_diff = (s_df['sofa_time'] - s_df['intime']).dt.total_seconds() / 3600
    else:
        s_df['INTIME'] = pd.to_datetime(s_df['INTIME'], errors='coerce')
        s_df['sofa_time'] = pd.to_datetime(s_df['sofa_time'], errors='coerce')
        time_diff = (s_df['sofa_time'] - s_df['INTIME']).dt.total_seconds() / 3600

    mask = (time_diff > min_hours) & (time_diff <= max_hours)
    s_df = s_df[mask]

    # Merge
    if list(s_df.columns) != list(o_df.columns):
        raise ValueError("Two files have different columns or order.")
    
    merged_df = pd.concat([s_df, o_df], ignore_index=True)

    # Filter with missing threshold
    cols = [col for col in merged_df.columns if '_mean' in col]
    if not cols:
        raise ValueError(f"No columns match the filter _mean")
    filtered_df = merged_df[cols]
    na_counts = filtered_df.isna().sum(axis=1)
    merged_df = merged_df[na_counts <= missing_threshold]
    
    # Keep the least missing
    merged_df['_na_count'] = merged_df.isna().sum(axis=1)
    df_sorted = merged_df.sort_values(by='_na_count')
    if ver == 4:
        merged_df = df_sorted.drop_duplicates(subset='subject_id', keep='first').drop(columns=['_na_count'])
    else:
        merged_df = df_sorted.drop_duplicates(subset='SUBJECT_ID', keep='first').drop(columns=['_na_count'])

    if ver == 3:
        merged_df = merged_df.rename(columns={'SUBJECT_ID': 'subject_id', 'ICUSTAY_ID': 'stay_id', 'INTIME': 'intime'})

    merged_df.to_csv(output_file, index=False)
    print(f"Done, output {len(merged_df)} rows to: {output_file}")

def cohort_filter_first(input_file, ver, min_hours, max_hours, missing_threshold, output_file):
    """
    Filter cohorts based on their SOFA time and number of missing columns.
    The only difference with cohort_filter is:
    Only the first ICU admission are kept for one patient.

    Parameters:
    input_file (str)          : input CSV file of cohorts
    ver (int)                 : MIMIC version, 3 or 4
    min_hours (float)         : earliest SOFA time compared to ICU admission time in hours, 0 by default
    max_hours (float)         : latest SOFA time compared to ICU admission time in hours, 6 by default
    missing_threshold (float) : rows with missing columns more than this threshold will be removed, 6.8 by default, decimals recommended   
    output_file (str)         : output CSV file
    """
    if ver != 3 and ver != 4:
        raise ValueError("Bad version of MIMIC.")
    
    df = pd.read_csv(input_file)

    # Keep the first addmission
    if ver == 4:
        df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
        df_sorted = df.sort_values(by='intime')
        df = df_sorted.drop_duplicates(subset='subject_id', keep='first').copy()
    else:
        df['INTIME'] = pd.to_datetime(df['INTIME'], errors='coerce')
        df_sorted = df.sort_values(by='INTIME')
        df = df_sorted.drop_duplicates(subset='SUBJECT_ID', keep='first').copy()

    # Filter with missing threshold
    cols = [col for col in df.columns if '_mean' in col]
    if not cols:
        raise ValueError(f"No columns match the filter _mean")
    filtered_df = df[cols]
    na_counts = filtered_df.isna().sum(axis=1)
    df = df[na_counts <= missing_threshold]

    # Filter with time
    if ver == 4:
        df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
        df['sofa_time'] = pd.to_datetime(df['sofa_time'], errors='coerce')
        time_diff = (df['sofa_time'] - df['intime']).dt.total_seconds() / 3600
    else:
        df['INTIME'] = pd.to_datetime(df['INTIME'], errors='coerce')
        df['sofa_time'] = pd.to_datetime(df['sofa_time'], errors='coerce')
        time_diff = (df['sofa_time'] - df['INTIME']).dt.total_seconds() / 3600

    sepsis_mask = df['sepsis']

    mask = ((time_diff > min_hours) & (time_diff <= max_hours)) | (sepsis_mask == 0)
    df = df[mask]

    # Save
    if ver == 3:
        df = df.rename(columns={'SUBJECT_ID': 'subject_id', 'ICUSTAY_ID': 'stay_id', 'INTIME': 'intime'})

    df.to_csv(output_file, index=False)
    print(f"Done, output {len(df)} rows to: {output_file}")

if __name__ == "__main__":
    #cohort_filter('../sepsis3_data.csv', '../other_data.csv', ver=4, min_hours=0, max_hours=6, missing_threshold=17, output_file='../dataset.csv')
    cohort_filter_first('../all_data.csv', ver=4, min_hours=0, max_hours=6, missing_threshold=17, output_file='../first/dataset_first2.csv')
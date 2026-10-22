def add_single_column(count_csv, label_csv, output_csv, id_column, target_column):
    """
    Add a target column to count_csv based on id_column matching.
    If the target column already exists, only missing values are imputed.

    Parameters:
        count_csv (str)     : csv file to add column
        label_csv (str)     : csv file with id_column and target_column
        output_csv (str)    : output csv
        id_column (str)     : column to match, subject_id in most cases
        target_column (str) : column to add
    """
    import pandas as pd
    
    count_df = pd.read_csv(count_csv)
    
    label_df = pd.read_csv(label_csv)
    
    if target_column in count_df.columns:
        print(f"'{target_column}' existing, only missing values are imputed.")
        
        temp_df = count_df.merge(
            label_df[[id_column, target_column]],
            on=id_column,
            how='left',
            suffixes=('', '_new')
        )

        new_col = f"{target_column}_new"
        
        if new_col in temp_df.columns:
            mask = count_df[target_column].isna()
            count_df.loc[mask, target_column] = temp_df.loc[mask, new_col]
        
        merged_df = count_df
    else:
        merged_df = count_df.merge(
            label_df[[id_column, target_column]],
            on=id_column,
            how='left'
        )
    
    merged_df.to_csv(output_csv, index=False)
    print(f"Done, result output to {output_csv}")
    
    return merged_df

if __name__ == "__main__":
    add_single_column('../all_data.csv', '../omr_weight.csv', '../all_data.csv', 'subject_id', 'weight')
    add_single_column('../all_data.csv', '../../../hosp/patients.csv', '../all_data.csv', 'subject_id', 'gender')
    add_single_column('../all_data.csv', '../../../hosp/patients.csv', '../all_data.csv', 'subject_id', 'anchor_age')
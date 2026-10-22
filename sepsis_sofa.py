import pandas as pd
import numpy as np

def create_sepsis_sofa_label(data_csv, sofa_csv, output_csv):

    df = pd.read_csv(data_csv)

    df = df[['subject_id', 'stay_id', 'intime', 'sepsis']]

    sofa_df = pd.read_csv(sofa_csv)
    
    merged_df = df.merge(
        sofa_df[['stay_id', 'sofa_adm']],
        on='stay_id',
        how='left'
    )

    merged_df["sofa_adm"] = merged_df["sofa_adm"].fillna(0)

    merged_df['sepsis_sofa'] = np.where(merged_df['sofa_adm'] >= 2, 1, 0)

    merged_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    #create_sepsis_sofa_label('../dataset.csv', 'sofa_adm.csv', '../sepsis_sofa.csv')
    create_sepsis_sofa_label('../first/dataset_first.csv', 'sofa_adm.csv', '../first/sepsis_sofa_first.csv')
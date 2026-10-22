import pandas as pd
import math

def add_label_features_sorted(file_a, file_b, file_c, output_file, start_time, end_time, version='IV'):
    """
    Add features to cohorts based on subject_id and itemid. The observation window can be specified.
    Each indicator has 8 statistical features:
        lastvalue - the last observed value in the observation window
        lasttime - time delta between the last value was observed and the ICU admission time in hours
        freq - number of records collected in the observation window
        mean - mean value of records collected in the observation window
        std - standard deviation of records collected in the observation window
        min - minimum value of records collected in the observation window
        max - maximum value of records collected in the observation window
        sampled - if the indicator is collected. 1 means collected and 0 means not
    For example, out of an indicator, 'Hematocrit', columns like 'Hematocrit_lastvalue' will be added.
    As a result, 8 columns will be added for each indicator listed in item file,
    though only 4 are kept in the final design.

    Parameters：
        file_a (str)     : cohort CSV file to add columns
        file_b (str)     : labevent file
        file_c (str)     : CSV used to provide item IDs
        output_file (str): cohort CSV file with columns added
        start_time (int) : hours from ICU admission to the observation window starts
        end_time (int)   : hours from ICU admission to the observation window ends
        version (str)    : deprecated, must be 'IV'
    """

    if version != 'III' and version != 'IV':
        print("No such version of MIMIC!")
        return None
    
    if version == 'III':
        item = 'ITEMID'
        ctime = 'CHARTTIME'
        vn = 'VALUE'
        stay_id_col='SUBJECT_ID'
        time_col='INTIME'
    else:
        item = 'itemid'
        ctime = 'charttime'
        vn = 'value'
        stay_id_col='subject_id'
        time_col='intime'

    df_a = pd.read_csv(file_a, parse_dates=[time_col])
    df_b = pd.read_csv(file_b, parse_dates=[ctime])
    df_c = pd.read_csv(file_c)

    if not df_a[stay_id_col].is_monotonic_increasing:
        raise ValueError(f"`df_a` must be sorted by {stay_id_col}")
    if not (df_b.sort_values([stay_id_col, ctime]).index == df_b.index).all():
        raise ValueError("df_b must be sorted by both 'subject_id' and 'charttime'")

    lasttime_blank = 2 * start_time - end_time

    if version == 'III':
        itemid_to_label = dict(zip(df_c['ITEMID'], df_c['label']))
    else:
        itemid_to_label = dict(zip(df_c['itemid'], df_c['label']))
    valid_itemids = set(itemid_to_label.keys())

    for label in df_c['label'].unique():
        df_a[f'{label}_lastvalue'] = pd.NA
        df_a[f'{label}_lasttime'] = pd.NA
        df_a[f'{label}_freq'] = 0
        df_a[f'{label}_mean'] = pd.NA
        df_a[f'{label}_std'] = pd.NA
        df_a[f'{label}_min'] = pd.NA
        df_a[f'{label}_max'] = pd.NA
        df_a[f'{label}_sampled'] = 0

    a_rows = df_a.to_dict('records')
    b_rows = df_b.to_dict('records')

    i = j = 0
    n, m = len(a_rows), len(b_rows)

    while i < n and j < m:
        a_row = a_rows[i]
        a_sid = a_row[stay_id_col]
        a_time = a_row[time_col]

        stats = {}

        while j < m and b_rows[j][stay_id_col] < a_sid:
            j += 1
        start_j = j

        while j < m and b_rows[j][stay_id_col] == a_sid:
            b_row = b_rows[j]
            itemid = b_row[item]
            if itemid not in valid_itemids:
                j += 1
                continue

            label = itemid_to_label[itemid]
            charttime = b_row[ctime]
            val = b_row[vn]
            delta = (charttime - a_time).total_seconds() / 3600

            if start_time <= delta < end_time:
                if label not in stats:
                    stats[label] = {
                        'lastval': val,
                        'lastdelta': delta,
                        'count': 1,
                        'mean': val,
                        'M2': 0.0,  # 用于 std
                        'min': val,
                        'max': val
                    }
                else:
                    s = stats[label]
                    if delta > s['lastdelta']:
                        s['lastval'] = val
                        s['lastdelta'] = delta

                    s['count'] += 1
                    delta_val = val - s['mean']
                    s['mean'] += delta_val / s['count']
                    s['M2'] += delta_val * (val - s['mean'])  # 在线更新 M2
                    s['min'] = min(s['min'], val)
                    s['max'] = max(s['max'], val)

            j += 1

        for label, s in stats.items():
            df_a.at[i, f'{label}_lastvalue'] = s['lastval']
            df_a.at[i, f'{label}_lasttime'] = s['lastdelta']
            df_a.at[i, f'{label}_freq'] = s['count']
            df_a.at[i, f'{label}_mean'] = s['mean']
            df_a.at[i, f'{label}_min'] = s['min']
            df_a.at[i, f'{label}_max'] = s['max']
            df_a.at[i, f'{label}_sampled'] = 1
            if s['count'] > 1:
                df_a.at[i, f'{label}_std'] = math.sqrt(s['M2'] / (s['count'] - 1))

        for label in df_c['label'].unique():
            if df_a.at[i, f'{label}_freq'] == 0:
                df_a.at[i, f'{label}_lasttime'] = lasttime_blank

        i += 1
        j = start_j

    df_a.to_csv(output_file, index=False)
    print("Done")
    return df_a

if __name__ == "__main__":
    #add_label_features_sorted('sepsis3_temp.csv', '../labevents.csv', 'bloodall.csv', '../sepsis3_data.csv', -12, 0)
    #add_label_features_sorted('other_temp.csv', '../labevents.csv', 'bloodall.csv', '../other_data.csv', -12, 0)
    add_label_features_sorted('all_temp.csv', '../labevents.csv', 'bloodall.csv', '../all_data.csv', -12, 0)
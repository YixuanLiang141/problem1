import pandas as pd
import numpy as np
import os

def filter_by_value_range_chunked(filter_file, item_file, output_path, chunksize=100000):
    """
    Filter labevents.csv based on the strategy in https://arxiv.org/abs/1703.07771.
    Itemids not in item_file are dropped.
    Create a new filtered CSV file and an outlier file.
    For example, if you name the output file as "1.csv",
    then another file "1_out_of_range.csv" is created.
    Only outlier rows are included, not those between valid and outlier.
    Records marked as unreliable in MIMIC-IV (value is ____) are considered outlier.

    Parameters:
        filter_file (str): CSV file to filter
        item_file (str)  : CSV used to provide item IDs and ranges
        output_file (str): CSV file after filtered
        chunksize (int)  : No modification is needed
    """

    # Read item file
    df_b = pd.read_csv(item_file)
    # Legacy code for compatibility with MIMIC-III
    df_b_columns = {c.lower(): c for c in df_b.columns}
    # expect ITEMID, min, max, minout, maxout
    # normalize to lowercase keys
    df_b = df_b.rename(columns={v: k.lower() for k, v in df_b_columns.items()})

    # Ensure item id column name
    if 'itemid' not in df_b.columns:
        raise KeyError("threshold table must have ITEMID column")

    # Build threshold DataFrame indexed by itemid (vectorized join later)
    thresholds = df_b[['itemid']].copy()
    for col in ('min', 'max', 'minout', 'maxout'):
        thresholds[col] = df_b.get(col, np.nan)

    thresholds = thresholds.set_index('itemid')

    # read file A in chunks
    itemid_col = 'itemid'
    valuenum_col = 'value'

    # If user asked to write files incrementally:

    base, ext = os.path.splitext(output_path)
    filtered_path = output_path
    out_path = base + "_out_of_range" + ext
    # remove existing files if any
    for p in (filtered_path, out_path):
        if os.path.exists(p):
            os.remove(p)
    write_header = True
    write_header_out = True

    # Read using chunksize
    read_csv_kwargs = {'chunksize': chunksize, 'low_memory': False}

    total_rows = 0
    kept_rows = 0
    out_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(filter_file, **read_csv_kwargs)):
        total_rows += len(chunk)

        # normalize chunk column names (some files may be mixed case)
        if itemid_col not in chunk.columns:
            # try lowercase alternative
            if itemid_col.lower() in chunk.columns:
                chunk = chunk.rename(columns={itemid_col.lower(): itemid_col})
            elif itemid_col.upper() in chunk.columns:
                chunk = chunk.rename(columns={itemid_col.upper(): itemid_col})
            else:
                raise KeyError(f"{itemid_col} not found in file A columns")

        if valuenum_col not in chunk.columns:
            if valuenum_col.lower() in chunk.columns:
                chunk = chunk.rename(columns={valuenum_col.lower(): valuenum_col})
            elif valuenum_col.upper() in chunk.columns:
                chunk = chunk.rename(columns={valuenum_col.upper(): valuenum_col})
            else:
                raise KeyError(f"{valuenum_col} not found in file A columns")

        # Merge thresholds into chunk by itemid (left join)
        # keep original index to preserve rows
        merged = chunk.merge(thresholds, how='left', left_on=itemid_col, right_index=True, suffixes=('', '_thr'))

        # create masks
        val = pd.to_numeric(merged[valuenum_col], errors="coerce")

        is_na = val.isna()

        # in normal range?
        # condition: (min is na or val >= min) & (max is na or val <= max)
        min_col = merged['min']
        max_col = merged['max']
        minout_col = merged['minout']
        maxout_col = merged['maxout']

        # rows to drop (not in threshold table)
        no_threshold = min_col.isna() & max_col.isna() & minout_col.isna() & maxout_col.isna()

        in_normal = (~is_na) & ((min_col.isna() | (val >= min_col)) & (max_col.isna() | (val <= max_col))) & (~no_threshold)

        # in extended (outside normal but inside minout/maxout)
        in_extended = (~is_na) & ((minout_col.isna() | (val >= minout_col)) & (maxout_col.isna() | (val <= maxout_col))) & (~no_threshold)

        keep_mask = in_normal

        # rows that are in extended but not normal -> need adjustment
        adjust_mask = in_extended & (~in_normal) & (~no_threshold)

        # rows to drop (out of extended range) or NaN valuenum
        drop_mask = (~is_na) & (~in_extended) & (~in_normal) & (~no_threshold)  # value outside extended range
        #nan_mask = is_na & (~no_threshold)  # missing valuenum and threshold exists -> drop

        out_mask = drop_mask

        # prepare adjusted values for adjust_mask
        adjusted_val = val.copy()

        # if below min and min exists -> set to min
        below_min = adjust_mask & min_col.notna() & (val < min_col)
        adjusted_val[below_min] = min_col[below_min]

        # if above max and max exists -> set to max
        above_max = adjust_mask & max_col.notna() & (val > max_col)
        adjusted_val[above_max] = max_col[above_max]

        # create output DataFrames for this chunk
        # filtered: keep_mask rows + adjust_mask rows with adjusted values applied
        filtered_chunk = merged[keep_mask | adjust_mask].copy()

        if not filtered_chunk.empty and adjust_mask.any():
            # apply adjusted values to valuenum_col for rows that were adjusted
            idxs = adjusted_val.index[adjust_mask]
            filtered_chunk.loc[idxs, valuenum_col] = adjusted_val.loc[idxs]

        out_chunk = merged[out_mask].copy()

        # drop the threshold columns before writing/returning to match original schema
        for thr_col in ('min', 'max', 'minout', 'maxout'):
            if thr_col in filtered_chunk.columns:
                filtered_chunk.drop(columns=[thr_col], inplace=True)
            if thr_col in out_chunk.columns:
                out_chunk.drop(columns=[thr_col], inplace=True)

        # write or accumulate
        if not filtered_chunk.empty:
            filtered_chunk = filtered_chunk[['subject_id','itemid','charttime','value']]
            filtered_chunk.to_csv(filtered_path, mode='a', header=write_header, index=False)
            write_header = False
            kept_rows += len(filtered_chunk)
        if not out_chunk.empty:
            out_chunk = out_chunk[['subject_id','itemid','charttime','value']]
            out_chunk.to_csv(out_path, mode='a', header=write_header_out, index=False)
            write_header_out = False
            out_rows += len(out_chunk)


        # optional: progress print
        print(f"Chunk {chunk_idx}: rows={len(chunk)} kept={len(filtered_chunk)} out_of_range={len(out_chunk)}")

    print(f"Total rows processed: {total_rows}")
    print(f"Total kept (including adjusted): {kept_rows}")
    print(f"Total out_of_range: {out_rows}")


    return filtered_path, out_path

def sort_csv_by_two_col(input_file, output_file, col1, col2):
    """
    Sort CSV by two columns.

    Parameters:
    input_file (str)    : input CSV file
    output_file (str)   : output CSV file
    col1 (str)          : column 1 to sort
    col2 (str)          : column 2 to sort
    """
    df = pd.read_csv(input_file)
    df.sort_values(by=[col1, col2], inplace=True)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    filter_by_value_range_chunked(
        filter_file = '../../../hosp/labevents.csv',
        item_file = 'bloodall.csv',
        output_path = '../labevents.csv'
    )

    sort_csv_by_two_col(
        input_file = "../labevents.csv",
        output_file = "../labevents.csv",
        col1 = "subject_id",
        col2 = "charttime"
    )


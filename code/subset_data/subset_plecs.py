# Training models on whole datasets is computational intensive and requires more storage.
# Here we assume the data table has been subsetted (e.g. with subset_tanle.sh) and we want to subset the PLEC .npy file accordingly.
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def subset_plecs(subset_table,npy_file,output_file="",column_name="PLEC_index"):
    """
    Subsets the PLEC fingerprint .npy dataset according to a table with column of indices.

    Parameters
    -------------------------------------------------------------
    subset_table: str
        Table containing a column with PLEC indices.
    npy_file: str
        path/to/numpy_file containing the PLEC fingerprints corresponding the table indices.
    output_file: str
        path/to/directory/output.npy location of output.
        By default it takes the directory of the .npy file and appends "_subset".
    column_name: str
        Column containing the PLEC indices. 
        By default it is "PLEC_index".
        
    Output
    -------------------------------------------------------------
    Creates a subsetted .npy file with only the PLEC fingerprints relevant to the subset table. Also reindexes the subset table PLEC indices to match.
    """

    df=pd.read_csv(subset_table)
    df = df.sort_values(by=column_name)
    try:
        unique_indices = np.array(df[column_name].unique())
        df[column_name] = range(len(df))
        reindexed_table=f"{subset_table.split('.csv')[0]}_reindexed.csv"
        df.to_csv(reindexed_table, index=False)
        plec_rows=len(df)
    except Exception as e:
        print(f"Error occured: {str(e)}")
    finally:
        del df
        print(f'df object has been deleted to save up RAM. The reindexed has been subset table has been saved to {reindexed_table}.')
    
    map = np.lib.format.open_memmap(npy_file, mode="r")
    plec_columns=map.shape[1]
    plec_dtype=map.dtype
    
    print('Subsetting...')
    if output_file=="":
        output_file=f"{npy_file.split('.npy')[0]}_subset.npy"
    
    subset_memmap=np.lib.format.open_memmap(output_file, mode='w+', dtype=plec_dtype, shape=(plec_rows,plec_columns))        
    with tqdm(total=len(unique_indices)) as pbar:
        for i,idx in enumerate(unique_indices):
            subset_memmap[i]=map[idx]
            pbar.update(1)
    
    subset_memmap.flush()

    print(f"Subset .npy file has been saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subsets the PLEC fingerprint .npy dataset according to a table with column of indices.')
    parser.add_argument('--subset_table', required=True, type=str,
                        help='Table containing a column with PLEC indices.')
    parser.add_argument('--npy_file', required=True, type=str,
                        help='path/to/numpy_file containing the PLEC fingerprints corresponding the table indices.')
    parser.add_argument('--output_file', required=False, type=str, default='',
                        help='path/to/directory/output.npy location of output. By default it takes the directory of the .npy file and appends _subset.')
    parser.add_argument('--column_name', required=False, type=str, default='PLEC_index',
                        help='Column containing the PLEC indices. By default it is PLEC_index.')
    args = parser.parse_args()
    
    subset_plecs(args.subset_table, args.npy_file, args.output_file, args.column_name)


    

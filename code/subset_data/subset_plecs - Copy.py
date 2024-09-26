# Training models on whole datasets is computational intensive and requires more storage.
# Here we assume the data table has been subsetted (e.g. with subset_tanle.sh) and we want to subset the PLEC .npy file accordingly.
# Use chunk_size parameter for time-memory trade off, because converting python lists to arrays is memory intensive.
# 100 = 5 min, 10000=11 min 1000=8 min 1 = 5 min
import pandas as pd
import numpy as np
from tqdm import tqdm

def subset_plecs(subset_table,npy_file,output_file="",column_name="PLEC_index",chunk_size=5000):
    """
    Subsets the PLEC fingerprint .npy dataset according to a table with column of indices.

    Parameters
    -------------------------------------------------------------
    subset_table: .csv file
        Table containing a column with PLEC indices.
    npy_file: str
        path/to/numpy_file containing the PLEC fingerprints corresponding the table indices.
    output_file: str
        path/to/directory/output.npy location of output.
        By default it takes the directory of the .npy file and appends "_subset".
    column_name: str
        Column containing the PLEC indices. 
        By default it is "PLEC_index".
    chunk_size: int
        Number of PLEC fingerprints written into np.array at a time, trades off computing speed for memory. Larger chunks reduce the overhead of operations but increase the amount of memory temporarily used during processing.
        By default it is 100
        
    Output
    -------------------------------------------------------------
    Creates a subsetted .npy file with only the PLEC fingerprints relevant to the subset table.
    """
    
    df=pd.read_csv(subset_table)
    try:
        unique_indices = np.array(sorted(df[column_name].unique()))
    except Exception as e:
        print(f"Error occured: {str(e)}")
    finally:
        plec_rows=len(df)
        del df
        print('df object has been deleted to save up RAM.')
    
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


subset_plecs(subset_table='D:/DNN_data/db_ML_table_diffdock_kinases_subset_10.csv', npy_file='D:/DNN_data/db_plecs_diffdock.npy', output_file='C:/Users/jiang/Documents/van_Westen/db_plecs_diffdock_subset_10.npy', column_name='PLEC_index')
    
    

# After subsetting the PLEC .npy file, the PLEC indexes are not correct anymore

df=pd.read_csv(subset_table)
df = df.sort_values(by=column_name)
try:
    unique_indices = np.array(sorted(df[column_name].unique()))
    df[column_name] = range(len(df))
    df.to_csv(f"{subset_table.split('.csv')[0]}_reindexed.csv", index=False)
except Exception as e:
    print(f"Error occured: {str(e)}")
finally:
    plec_rows=len(df)
    del df
    print('df object has been deleted to save up RAM.')

#!/bin/bash

# Define the input and output files
input_file="/d/DNN_data/db_ML_table_diffdock_kinases_copy.csv"
output_file="/d/DNN_data/db_ML_table_diffdock_kinases_subset_10.csv"

# Extract the header
header=$(head -n 1 $input_file)

# Extract unique 'accessions' values
awk -F, 'NR>1 {print $3}' $input_file | sort | uniq > unique_accessions.txt

# Function to process a partition
process_partition() {
  local partition=$1
  local partition_file="${partition}_partition.csv"
  local subset_file="${partition}_subset.csv"
  
  # Extract rows for the current partition, excluding the header
  awk -F, -v part="$partition" '$8 == part' $input_file | tail -n +2 > $partition_file
  
  # Process each unique 'accessions' value
  while read accession; do
    # Extract rows for the current 'accessions' value
    awk -F, -v acc="$accession" '$3 == acc' $partition_file > accession_partition.csv
    
    # Count the number of lines for the current 'accessions'
    accession_count=$(wc -l < accession_partition.csv)
    
    # Calculate 10% of the number of lines
    subset_count=$((accession_count / 10))
    
    # Shuffle and select 10% of the lines
    shuf accession_partition.csv | head -n $subset_count >> $subset_file
    
    # Clean up the intermediate file
    rm accession_partition.csv
  done < unique_accessions.txt
  
  # Clean up the partition file
  rm $partition_file
}

# Process 'test' and 'train' partitions
process_partition "test"
process_partition "train"

# Initialize the subset file with the header
echo "$header" > $output_file

# Combine the subsets into the output file
cat test_subset.csv train_subset.csv >> $output_file

# Clean up the subset files
rm test_subset.csv train_subset.csv

# Clean up the unique accessions file
rm unique_accessions.txt

echo "Subset created in $output_file"

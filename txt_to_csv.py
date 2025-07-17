import csv

# Correct file paths (use raw strings to avoid backslash issues)
input_file = r'Your text file path here'
output_file = r'Your output CSV file path here'

# Open and read the text file
with open(input_file, 'r') as txt_file:
    lines = txt_file.readlines()

# Write to a new CSV file
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header
    header = lines[0].strip().split(';')
    csv_writer.writerow(header)

    # Write data rows
    for line in lines[1:]:
        row = line.strip().split(';')
        csv_writer.writerow(row)

print(f"Text file has been successfully converted to CSV: {output_file}")
# The code reads a text file with semicolon-separated values and writes it to a CSV file with comma-separated values.
# It handles the file paths correctly and ensures that the output file is created in the specified location.
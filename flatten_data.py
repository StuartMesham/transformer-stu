import argparse
import csv

parser = argparse.ArgumentParser(prog="Data Flattener")
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

with open(args.input_file) as f_in, open(args.output_file, "w") as f_out:
    reader = csv.reader(f_in, skipinitialspace=True)
    header = next(reader)
    original_index = header.index("original_text")
    reframed_index = header.index("reframed_text")

    for row in reader:
        f_out.write(row[original_index] + "\n")
        f_out.write(row[reframed_index] + "\n")

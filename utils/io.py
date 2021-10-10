"""
    I/O functions
"""
import csv
import pickle


# ------------------------------------------------------------------------------
#    Pickle
# ------------------------------------------------------------------------------
def load_pickle(datafile):
    data = None
    with open(datafile, 'rb') as infile:
        data = pickle.load(infile)
    return data

def save_pickle(data, datafile):
    with open(datafile, 'wb') as outfile:
        pickle.dump(data, outfile)
    return data


# ------------------------------------------------------------------------------
#    CSV
# ------------------------------------------------------------------------------
def load_csvfile(csvfile):
    data = []
    with open(csvfile, 'r') as infile:
        csv_reader = csv.reader(infile)
        for each_line in csv_reader:
            data.append(each_line)
    return data

def save_csvfile(data, csvfile):
    with open(csvfile, 'w') as outfile:
        csv_writer = csv.writer(outfile)
        for each_line in data:
            csv_writer.writerow(each_line)
    # done.

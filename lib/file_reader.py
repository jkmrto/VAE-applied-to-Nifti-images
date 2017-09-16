import csv


def read_csv_as_list_of_dictionaries(path_file):

    file = open(path_file)

    reader = csv.DictReader(file)
    list_rows = []
    for row in reader:
        list_rows.append(row)

    return list_rows

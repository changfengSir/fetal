import numpy as np
def read_csv_data(path):
    import csv
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_data=[]
        for row in reader:
            csv_data.append(row)
        csv_data_np = np.array(csv_data)
    return csv_data_np


data = read_csv_data('./balanced_test.csv')
print(data[0,1])
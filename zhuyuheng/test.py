import csv
import numpy as np

# with open('data.txt', 'rb') as csvfile:
#     file = csvfile.read().decode('utf-8')
#     file1 = file.split('\n')
#     print(file1[-1])
          # .split('\t'))
    # with open('data1.rcsv','wb') as f:
    #     ff = csv.writer(f)
    #     ff.writerow(file)
    # reader = csv.reader(csvfile)
    # column = [row[0] for row in reader]
    # print(column)

# a = numpy.array([1,2,3])
# print(a)

# n_out = np.random.rand(1,3,1)
# print(n_out.shape)
# a = np.array([[[1]]])
# print(a.shape)

a = np.array(['1.0'])
b = a.astype('float')
print(b)
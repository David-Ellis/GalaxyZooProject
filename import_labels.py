# imports labels for all questions from csv file 'galaxy_zoo_labels.csv'
# headers and labels are stored separetely in np.arrays of type float

import csv
import numpy as np

filename = 'galaxy_zoo_labels.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    labels = np.array(list(reader)).astype(float)
f.close()
    
#print(headers)
#print(data.shape)





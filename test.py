import numpy as np
lst = [8,5,7,5,9,1]
data = ','.join([str(i) for i in lst])
with open('sample.csv', 'w', newline='') as file:
    file.write(data)  
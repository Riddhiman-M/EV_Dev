import csv
import numpy as np
import pandas as pd

store = pd.read_excel(r'C:/Users/Riddhiman Moulick/IIT_Kharagpur/pythonProject/Electric_Vehicle/TrafficData1.xlsx')

num = store["Num"].to_numpy(dtype='float')
mu = np.mean(num)
dev = np.std(num)

with open('NormalisedPoints.csv', 'w') as file:
    cwriter = csv.writer(file)
    header = ['Num']
    cwriter.writerow(header)
    for i in range(len(num)):
        if (num[i] - mu) / dev > 2:
            np.delete(num, i)
        else:
            arr = [num[i]]
            cwriter.writerow(arr)

file.close()

# num = 0.1 < (num - mu) / dev < 2

print((num-mu)/dev)

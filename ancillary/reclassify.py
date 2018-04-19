import os

file = 'HTRU_2.csv'
ofile = 'HTRU_2_inverse.csv'
new_data = []

with open(file, 'r') as f:
    data = f.read()   
    data = data.split('\n') 
    for i, item in enumerate(data):
        if item[-1] == '1':
            # print(item[-1])
            new_data.append(item[:len(item)-1] + '0')
        else:
            new_data.append(item[:len(item)-1] + '1')


with open(ofile, 'w') as w:
    for item in new_data:
        w.write(item + "\n")
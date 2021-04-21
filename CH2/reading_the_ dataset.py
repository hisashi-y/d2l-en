import os

os.makedirs(os.path.join('/Users/hisashi-y/d2l-en/CH2', 'data'), exist_ok = True)
data_file = os.path.join('/Users/hisashi-y/d2l-en/CH2', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

inputs, outputs = data_file.iloc[:, 0:2], data_file.iloc[:, 2]

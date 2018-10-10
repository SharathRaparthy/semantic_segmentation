import pandas as pd
import os


path = '/home/sharath/gym-duckietown/dataset_1'

list = []

for i in os.listdir(path):
    list.append(i)

data_frame = pd.DataFrame(list, columns=['filename'])
data_frame.to_csv('data.csv', index = None)
data = pd.read_csv('data.csv')
print(data)

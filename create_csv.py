import pandas as pd
import os


path = '/home/sharath/Downloads/MSRC_ObjCategImageDatabase_v2/GroundTruth/'
list = []


for i in os.listdir(path):
    list.append(i)

data_frame = pd.DataFrame({'FileName': list})
data_frame.to_csv('/home/sharath/semantic_segmentation/data.csv', index = None)

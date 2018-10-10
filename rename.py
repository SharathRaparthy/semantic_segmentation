import os
path = '/home/sharath/Downloads/MSRC_ObjCategImageDatabase_v2/Images/'
files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path,file),os.path.join(path, str(i)+'.bmp'))
    i = i + 1

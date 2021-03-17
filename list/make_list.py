import numpy as np
import os
import glob

root_path = '/Users/Satvik/Desktop/UCLA/CS_260/project/I3d/DeepMIL/RGB/'
dirs = os.listdir(root_path)
print(dirs)

f = open('ucf-i3d.list', 'w+')
normal = []
for dir in dirs:
  files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
  print(files)
  for file in files:
    if 'Normal_' in file:
      normal.append(file)

for file in normal:
  newline = file+'\n'
  f.write(newline)

root_path = '/Users/Satvik/Desktop/UCLA/CS_260/project/I3d/DeepMIL/RGB/RGB0'
dirs = os.listdir(root_path)
print(dirs)
anomaly = []
for dir in dirs:
  files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
  print(files)
  for file in files:
    if 'x264' in file:
      anomaly.append(file)

for file in anomaly:
  newline = file+'\n'
  f.write(newline)

f = open('ucf-i3d-test.list', 'w+')
root_path = '/Users/Satvik/Desktop/UCLA/CS_260/project/I3d/DeepMIL/TestRGB'
dirs = os.listdir(root_path)
print(dirs)
test = []
for dir in dirs:
  files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
  print(files)
  for file in files:
    if 'x264' in file:
      test.append(file)

for file in test:
  newline = file+'\n'
  f.write(newline)
# root_path = '/Users/Satvik/Desktop/UCLA/CS_260/project/I3d/DeepMIL/RGB/Training_Normal_Videos_Anomaly1'
# dirs = os.listdir(root_path)
# print(dirs)
# with open('ucf-c3d.list', 'w+') as f:
#     normal = []
#     for dir in dirs:
#         files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
#         for file in files:
#             if '__0' in file:  ## comments
#                 if 'Normal_' in file:
#                     normal.append(file)
#                 else:
#                     newline = file+'\n'
#                     f.write(newline)
#     for file in normal:
#         newline = file+'\n'
#         f.write(newline)
#
# root_path = '/Users/Satvik/Desktop/UCLA/CS_260/project/I3d/DeepMIL/TestRGB'
# dirs = os.listdir(root_path)
# print(dirs)
# with open('ucf-c3d-test.list', 'w+') as f:
#     normal = []
#     for dir in dirs:
#         files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
#         for file in files:
#             if '__0' in file:  ## comments
#                 if 'Normal_' in file:
#                     normal.append(file)
#                 else:
#                     newline = file+'\n'
#                     f.write(newline)
#     for file in normal:
#         newline = file+'\n'
#         f.write(newline)

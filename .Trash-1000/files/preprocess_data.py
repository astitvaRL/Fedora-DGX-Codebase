import os
import natsort



# set paths
DATA_ROOT = '/home/astitva/DATA/'
drawings_dir_path = os.path.join(DATA_ROOT, 'amateur_drawings')
labels_dir_path = os.path.join(DATA_ROOT, 'AD_SegMaps/labels_2k')

labels = natsort.natsorted(os.listdir(labels_dir_path))
for label in labels:
    print(label)

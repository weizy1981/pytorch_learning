import os
import sys
import shutil

train_labels = 'train_labels.csv'
test_labels = 'test_labels.csv'
img_folder = 'data'

def move_file(filename, folder, isTrain=True):
    filename = filename.strip()
    folder = folder.strip()
    if isTrain:
        mid_folder = 'train'
    else:
        mid_folder = 'test'
    path = img_folder + '/' + mid_folder + '/' + folder.strip()
    if not os.path.exists(path):
        os.mkdir(path)
    old_file = img_folder + '/' + filename
    new_file = path + '/' + filename
    if not os.path.exists(new_file):
        shutil.copyfile(old_file, new_file)
        print(mid_folder + ': move file ' + filename + ' to folder ' + folder)
        os.remove(old_file)

if __name__ == '__main__':
    # train dataset
    if not os.path.exists('data/train'):
        os.mkdir('data/train')
    with open(train_labels, 'r') as train_f:
        lines = train_f.readlines()

    for line in lines:
        filename, folder = line.split(',')
        move_file(filename, folder)

    # test dataset
    if not os.path.exists('data/test'):
        os.mkdir('data/test')
    with open(test_labels, 'r') as train_f:
        lines = train_f.readlines()

    for line in lines:
        filename, folder = line.split(',')
        move_file(filename, folder, isTrain=False)


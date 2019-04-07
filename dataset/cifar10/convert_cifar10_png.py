# coding: UTF-8

from __future__ import  absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
from PIL import Image

from reader import Cifar10Reader


def convert(datafile, basename, offset=0, count=1, save_dir='./data'):
    file = open('./' + basename + '.txt', 'w')

    reader = Cifar10Reader(datafile)
    stop = offset + count

    for index in range(offset, stop):
        image = reader.read(index)
        im = Image.fromarray(image.byte_array.astype(np.uint8))

        file_name='%s-%04d-%d' % (basename, index, image.label)
        file_name = file_name + ".png"

        with open(save_dir + '/' + file_name, mode='wb') as out:
            im.save(out, format='png')

        file.write('%s, %d\n' % (save_dir + '/' + file_name, image.label))

    reader.close()
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cifar-10 converter')
    parser.add_argument('--offset', type=int, default=None)
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--base_name', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    basename = args.base_name
    convert(args.data, basename, args.offset, args.count, args.save_dir)

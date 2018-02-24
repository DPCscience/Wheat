"""
Splits a given dataset into train, test and dev.
Also resizes the images to be a specified size,
default is 224x224, VGG's input size.
"""
import argparse
import random
import os

from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', default='Data/WHEAT/Binary/',
    help="Directory with the dataset"
)
parser.add_argument(
    '--output_dir', default='Data/WHEAT/Binary/256x256_Binary',
    help="Where to write the new data"
)
parser.add_argument(
    '--output_size', default='256', help="What size to resize the images to. \
    Defaut is 256x256"
)

def resize_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    try:
        image = Image.open(filename).convert('RGB')
        image = image.resize((size, size), Image.BILINEAR)
        image.save(os.path.join(output_dir, filename.split('/')[-1]))
    except (IOError, ZeroDivisionError) as e: # Some image_files are malformed, ignore
        print(e)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the \
    dataset at {}".format(args.data_dir)

    data_dir = os.path.join(args.data_dir, 'full_dataset')
    SIZE = int(args.output_size)

    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg')]

    # Split the images in 'full_dataset' into 80% train and 10%  10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(753)
    filenames.sort()
    random.shuffle(filenames)

    train_split = int(0.8 * len(filenames))
    train_filenames = filenames[:train_split]
    rest = filenames[train_split:]

    dev_split = int(0.5 * len(rest))
    dev_filenames = rest[dev_split:]
    test_filenames = rest[:dev_split]

    filenames = {
        'train': train_filenames,
        'dev':   dev_filenames,
        'test':  test_filenames
    }
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
            print("Processing {} data, saving preprocessed \
            data to {}".format(split, output_dir_split))
            for filename in tqdm(filenames[split]):
                resize_and_save(filename, output_dir_split, size=SIZE)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
    print("Done building dataset")

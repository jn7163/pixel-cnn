import os
import numpy as np
import PIL

DEFAULT_IMAGENET_PATH = '/media/NAS_RO/ILSVRC2012/Extracted/'
SIZE = 256

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """
        - data_dir is the location of extracted ILSVRC2012 train data
              on our system: DEFAULT_IMAGENET_PATH
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        if subset == 'train':
            data_path = os.path.join(data_dir, 'train')
            self.image_paths = [
                os.path.join(data_path, d, i)
                for d in os.listdir(data_path)
                for i in os.listdir(os.path.join(data_path, d))
            ]
            assert len(self.image_paths) == 1281167
        elif subset == 'test':
            data_path = os.path.join(data_dir, 'test')
            self.image_paths = [
                os.path.join(data_path, i)
                for i in os.listdir(data_path)
            ]
            assert len(self.image_paths) == 100000
        else:
            raise ValueError('invalid subset: %s' % subset)
        
        self.image_paths = np.array(self.image_paths)
        fixed_rng = np.random.RandomState(1)
        fixed_rng.shuffle(self.image_paths)
        if subset == 'train':
            self.image_paths = self.image_paths[:100000]
        else:
            self.image_paths = self.image_paths[:5000]

        self.batch_size = batch_size
        self.shuffle = shuffle
        if return_labels:
            raise ValueError('return_labels not supported')
        

        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return (SIZE, SIZE, 3)

    def get_num_labels(self):
        raise NotImplementedError

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size
            
        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            self.rng.shuffle(self.image_paths)

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.image_paths.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = np.zeros((n, 256, 256, 3), dtype=np.uint8)
        for i, v in enumerate(self.image_paths[self.p : self.p + n]):
            x[i,:,:,:] = crop(v, randomize=self.shuffle)
        self.p += n
        
        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
    
# expects a path to a BW or color image
def crop(image_path, randomize):
    img = PIL.Image.open(image_path)
    if img.width < SIZE or img.height < SIZE:
        ratio = max(SIZE/img.width, SIZE/img.height)
        scaled_h = int(img.height * ratio)
        scaled_w = int(img.width * ratio)
        nw = max(SIZE, scaled_w)
        nh = max(SIZE, scaled_h)
        img = img.resize((nw, nh))
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    h, w, _ = img.shape
    if randomize:
        hstart = np.random.randint(h - SIZE + 1)
        wstart = np.random.randint(w - SIZE + 1)
    else:
        # center crop
        hstart = int((h - SIZE) / 2)
        wstart = int((w - SIZE) / 2)
    hend = hstart + SIZE
    wend = wstart + SIZE
    return img[hstart:hend, wstart:wend, :]

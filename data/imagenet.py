import os
import numpy as np
import PIL
import multiprocessing
import multiprocessing.dummy as dummy

DEFAULT_IMAGENET_PATH = '/media/NAS_RO/ILSVRC2012/Extracted/'
SIZE = 256

class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, buffer_size=None, dtype=np.uint8):
        """
        - data_dir is the location of extracted ILSVRC2012 train data
              on our system: DEFAULT_IMAGENET_PATH
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.dtype = dtype
        if dtype not in [np.uint8, np.float32]:
            raise ValueError('invalid dtype')
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
            # self.image_paths = self.image_paths[:1000000]
            pass
        else:
            self.image_paths = self.image_paths[:10000] # don't want to waste time on this

        self.batch_size = batch_size
        self.shuffle = shuffle
        if return_labels:
            raise ValueError('return_labels not supported')


        self.rng = np.random.RandomState(1) if rng is None else rng

        if buffer_size is None:
            buffer_size = 5 * batch_size
        if buffer_size < batch_size:
            raise ValueError('bad buffer size')
        self._buffer_size = buffer_size

        self._launch_workers()

    def _launch_workers(self):
        workers = []
        input_queue = multiprocessing.Queue(self._buffer_size)
        self._output_queue = multiprocessing.Queue(self._buffer_size)
        p = dummy.Process(target=pusher, args=(self.image_paths, input_queue, self._output_queue, self.shuffle))
        p.start()
        workers.append(p)
        for i in range(multiprocessing.cpu_count()):
            p = dummy.Process(target=worker, args=(input_queue, self._output_queue))
            p.start()
            workers.append(p)

    def get_observation_size(self):
        return (SIZE, SIZE, 3)

    def get_num_labels(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def reset(self):
        pass # ignore

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on intermediate iterations fetch the next batch
        if self.dtype == np.uint8:
            x = np.zeros((n, 256, 256, 3), dtype=np.uint8)
        else:
            x = np.zeros((n, 256, 256, 3), dtype=np.float32)
        for i in range(n):
            img = self._output_queue.get()
            if img is None:
                raise StopIteration
            if self.dtype == np.uint8:
                x[i,:,:,:] = img
            else:
                x[i,:,:,:] = img / 255.0

        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)

def pusher(image_paths, input_queue, output_queue, shuffle):
    rng = np.random.RandomState(1)
    while True:
        if shuffle:
            rng.shuffle(image_paths)
        for path in image_paths:
            input_queue.put((path, shuffle))
        input_queue.put(None) # sloppy epochs

def worker(input_queue, output_queue):
    while True:
        inp = input_queue.get()
        if inp is None:
            output_queue.put(None)
            continue
        path, randomize = inp
        cropped = crop(path, randomize)
        output_queue.put(cropped)

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

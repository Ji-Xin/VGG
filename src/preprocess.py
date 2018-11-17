import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

def mnist(augment=False):
    for fname in ["mnist.train", "mnist.test"]:
        with open(fname) as f:
            x = []
            y = []
            for line in f.readlines()[1:]:
                a = line.strip().split(',')
                y.append(int(a[0]))
                x_raw = np.reshape(
                    np.array(list(map(lambda x: float(x), a[1:])), dtype=np.int32), [28, 28])
                x_im = reshape(x_raw, 32)
                x.append(x_im)
                if augment:
                    x_im_aug = rotate(blur(x_im, np.random.randint(7)), np.random.randint(-45, 46))
                    x.append(x_im_aug)
                    y.append(int(a[0]))
            np.save(fname+".x"+(".aug" if augment else "")+".npy", np.array(x))
            np.save(fname+".y"+(".aug" if augment else "")+".npy", np.array(y))
        if augment:
            break # no need to change testset


def reshape(raw, new_dim):
    image = Image.fromarray(raw)
    newimage = image.resize([new_dim, new_dim])
    return np.array(newimage)

def rotate(raw, angle):
    image = Image.fromarray(raw)
    newimage = image.rotate(angle)
    return np.array(newimage)

def blur(raw, radius):
    new = gaussian_filter(raw, radius)
    return new



# trial()
if __name__=="__main__":
    mnist()
    mnist(augment=True)

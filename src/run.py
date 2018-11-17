import numpy as np
import argparse

from plot import plot, plot_compromised
from model import VGG11
from preprocess import rotate, blur

from keras.models import load_model



#################################### parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", dest="data_dir", required=True, action="store",
    help="path of dataset")
parser.add_argument("--output_dir", dest="output_dir", required=True, action="store",
    help="path of recording training history and saving final model")
parser.add_argument("--fig_dir", dest="fig_dir", required=True, action="store",
    help="path of figures")
parser.add_argument("--marker", dest="marker", required=True, action="store",
    help="marker of experiments")
parser.add_argument("--l2_reg", dest="l2_reg", action="store_true",
    help="using l2 regularization")
parser.add_argument("--test", dest="test", action="store_true",
    help="test only")
parser.add_argument("--aug", dest="aug", action="store_true",
    help="training data augmentation")
args = parser.parse_args()



###################################### build model
model = VGG11(l2_reg=args.l2_reg).model


#################################### load data
if not args.test:
    if args.aug:
        train_x = np.load(args.data_dir+"/mnist.train.x.aug.npy")
        train_y = np.load(args.data_dir+"/mnist.train.y.aug.npy")
    else:
        train_x = np.load(args.data_dir+"/mnist.train.x.npy")
        train_y = np.load(args.data_dir+"/mnist.train.y.npy")
test_x = np.load(args.data_dir+"/mnist.test.x.npy")
test_y = np.load(args.data_dir+"/mnist.test.y.npy")


if not args.test:
    train_x = np.expand_dims(train_x, 3)/255.0
    test_x = np.expand_dims(test_x, 3)/255.0



################################### train model / test model
if not args.test:
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
        shuffle=True, epochs=20, batch_size=256)
    print(model.evaluate(test_x, test_y, batch_size=1000))

    np.save(args.output_dir+"/"+args.marker+".npy", history.history)
    model.save(args.output_dir+"/"+args.marker+".h5")
    plot(history.history, args.fig_dir, args.marker)

else:
    model = load_model(args.output_dir+"/"+args.marker+".h5")
    rotate_result = []
    blur_result = []
    for angle in range(-45, 50, 5):
        new_test_x = []
        for i in range(len(test_x)):
            new_test_x.append(rotate(test_x[i], angle))
        new_test_x = np.expand_dims(np.array(new_test_x), 3)/255.0
        result = model.evaluate(new_test_x, test_y, batch_size=1000)
        rotate_result.append([angle, result[0], result[1]])
    print(rotate_result)

    for radius in range(0, 7, 1):
        new_test_x = []
        for i in range(len(test_x)):
            new_test_x.append(blur(test_x[i], radius))
        new_test_x = np.expand_dims(np.array(new_test_x), 3)/255.0
        result = model.evaluate(new_test_x, test_y, batch_size=1000)
        blur_result.append([radius, result[0], result[1]])
    print(blur_result)
    np.save(args.output_dir+"/"+args.marker+"_rotate.npy", rotate_result)
    np.save(args.output_dir+"/"+args.marker+"_blur.npy", blur_result)

    plot_compromised(rotate_result, "angle", args.fig_dir, args.marker+"_rotate")
    plot_compromised(blur_result, "radius", args.fig_dir, args.marker+"_blur")

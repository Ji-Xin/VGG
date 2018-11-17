import numpy as np
import matplotlib.pyplot as plt


def plot(history, dir, marker):
    plt.subplot(1, 2, 1)
    plt.yscale("logit")
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.yscale("log")
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(dir+"/"+marker+".pdf")
    plt.close()


def plot_compromised(content, xlabel, dir, marker):
    x = [c[0] for c in content]
    y = [c[2] for c in content]
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.savefig(dir+"/"+marker+".pdf")
    plt.close()

if __name__=="__main__":
    history = np.load("output/raw.npy").item()
    plot(history, "figs", "raw_log")
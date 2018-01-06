import matplotlib.pyplot as plt
import argparse
import numpy as np


def parse_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',')
    epoch = data[:, 0]
    batch = data[:, 1]
    cost = data[:, 2]
    test_cost = data[:, 3]
    test_accuracy = data[:, 4]
    return epoch, batch, cost, test_cost, test_accuracy


def plot_cost(filepath, title, output):
    epoch, _, cost, test_cost, _ = parse_csv(filepath)

    plt.ylabel('-IOU (Jaccard)')
    plt.xlabel('Epoch')

    plt.plot(epoch, cost, 'r-', label='train')
    plt.plot(epoch, test_cost, 'b-', label='test')
    plt.legend()

    if title:
        plt.title(title)
    if output:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()


def plot_accuracy(filepath, title, output):
    epoch, _, _, _, test_accuracy = parse_csv(filepath)

    plt.ylabel('Test accuracy (%)')
    plt.xlabel('Epoch')

    plt.plot(epoch, test_accuracy, 'r-')

    if title:
        plt.title(title)
    if output:
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--file', dest='file')
    parser.add_argument('--title', dest='title')
    parser.add_argument('--output', dest='output')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    command_dict = {'cost': plot_cost, 'accuracy': plot_accuracy}
    command_dict[args.command](args.file, args.title, args.output)

from matplotlib import pyplot as plt


def axis3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax

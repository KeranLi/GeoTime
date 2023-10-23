import matplotlib.pyplot as plt

def plot_line(x, y1, y2, title, xlimit=None, ylimit=None):

    plt.plot(x, y1, label='Original')
    plt.plot(x, y2, label='Detrended')

    if xlimit:
        plt.xlim(xlimit)
    
    if ylimit:
        plt.ylim(ylimit)

    plt.xlabel('Age (Ma)')
    plt.ylabel('CAR(mg/cm2/kyr)')
    plt.title(title)
    plt.legend()
    plt.show()
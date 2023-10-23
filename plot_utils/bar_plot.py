import matplotlib.pyplot as plt

def plot_bar(x, y, title='Default',
             auto_axis=True,
             xlimit=None,
             ylimit=None):
    """
    @xlimit=[start,end]
    @ylimit=[start,end]
    """
    plot_x = x
    plot_y = y

    if not auto_axis and xlimit and ylimit:

        plt.plot(plot_x,plot_y)
        plt.xlim(xlimit)
        plt.ylim(ylimit)
        plt.title(title)
        plt.show()
    else:
        # 自动轴范围
        if x is not None and y is not None:
            plt.plot(plot_x,plot_y)
            plt.title(title)
            plt.show()

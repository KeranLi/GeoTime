import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_psd_decomposition(age, Y, frequency):
    """
    Visualize the PSD analysis results on diffrent frequencies

    Parameters:
    age (array-like): Geological time。
    Y_reconstructed (array-like): Original data or normalized signals。

    Returns:
    None
    """
    # Set up the main fig and sub-figs
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Plot the original sequence
    ax = fig.add_subplot(gs[0, :])
    ax.plot(age, Y, label='Detrend data')
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Normalized Signal')

    # Plot the PSD with different amounts of zero padding. This uses the entire time series at once
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.psd(Y, NFFT=len(Y), pad_to=len(Y), Fs=frequency)
    ax2.psd(Y, NFFT=len(Y), pad_to=len(Y) * 2, Fs=frequency)
    ax2.psd(Y, NFFT=len(Y), pad_to=len(Y) * 4, Fs=frequency)
    plt.title('Zero padding')

    # Plot the second sub-figs of PSD
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.psd(Y, NFFT=len(Y), pad_to=len(Y), Fs=frequency)
    ax3.psd(Y, NFFT=len(Y), pad_to=len(Y) * 2, Fs=frequency)
    ax3.psd(Y, NFFT=len(Y), pad_to=len(Y) * 4, Fs=frequency)
    plt.title('Block size')

    # Plot the third sub-figs of PSD
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.psd(Y, NFFT=len(Y), pad_to=len(Y), Fs=frequency)
    ax4.psd(Y, NFFT=len(Y), pad_to=len(Y) * 2, Fs=frequency)
    ax4.psd(Y, NFFT=len(Y), pad_to=len(Y) * 4, Fs=frequency)
    plt.title('Overlap')

    plt.show()
import matplotlib.pyplot as plt


# takes a text file and saves a histogram of text lengths
def plot_distribution(textfile, author):
    lengths = get_lengths(textfile)
    plot_figure(lengths, author)


# Return list of lengths for each line
def get_lengths(textfile):
    lengths = []
    for line in open(textfile):
        length = len(line)
        if length > 1:
            lengths.append(length)
    return lengths


# Plots figure
def plot_figure(lengths, author):
    plt.hist(lengths)
    plt.title("Tweet length distribution " + author)
    plt.xlabel("Text length")
    plt.ylabel("Frequency")
    fname = "visualizations/output/dist_" + author + ".png"
    plt.savefig(fname, format='png')

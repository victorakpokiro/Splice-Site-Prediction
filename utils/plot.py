from matplotlib import pyplot as plt


class Plot(object):
    """docstring for plot"""
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    @staticmethod
    def setup_graph(x_width=12, y_width=9, font_size=16):
        plt.figure(figsize=(x_width, y_width))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.yticks(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        #plt.tick_params(axis="both", which="both", bottom="on", top="off",
                    #labelbottom="on", left="on", right="off", labelleft="on")
        return plt

    @staticmethod
    def plot_time_series(x_data, y_data, fig_path_name, x_title, y_title, title, tags='b-',color_index=0):
        Plot.setup_graph()
        t = Plot.tableau20[color_index%len(Plot.tableau20)]
        t2 = tuple(ti/255 for ti in t)
        plt.plot(x_data, y_data,tags,color=t2)
        #plt.plot(x_data, y_data,tags,color=t2)
        plt.suptitle(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        #plt.legend(bbox_to_anchor=(0,1),loc="upper left")
        plt.savefig(fig_path_name, bbox_inches='tight')
        plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('empirical_results.csv')
    df['class_sep'] = df['class_sep'].astype(str)

    plt.figure(figsize=(8, 5))
    sns.set_style('darkgrid')
    plot = sns.lineplot(data=df, x='positive_weight', y='f1', hue='class_sep')
    plot.set(xscale='log')
    plt.xlabel('$ \\alpha $')
    plt.ylabel('$ F_1 $')
    plt.gca().get_legend().remove()
    plt.ylim((0, 1))
    # plt.show()
    plt.savefig('empirical_scores.png', dpi=144)
    plt.close()


if __name__ == '__main__':
    main()

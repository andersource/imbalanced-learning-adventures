import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    res = []

    for p in np.logspace(.3, 1.6, 15):
        beta = 1e-2
        alpha = np.logspace(0, 2.5, 100)

        x = (alpha * beta / (1 - beta + alpha * beta)) ** (1 / p)
        z = (1 - x ** p) ** (1 / p)

        fscore = 2 * beta * x / (beta * (x + z) + 1 - z)

        res.append(pd.DataFrame(dict(alpha=alpha, fscore=fscore, p=p)))

    df = pd.concat(res).reset_index(drop=True)
    df['p'] = df['p'].astype(str)

    plt.figure(figsize=(8, 5))
    sns.set_style('darkgrid')
    plot = sns.lineplot(data=df, x='alpha', y='fscore', hue='p')
    plot.set(xscale='log')
    plt.xlabel('$ \\alpha $')
    plt.ylabel('$ F_1 $')
    plt.gca().get_legend().remove()
    plt.ylim((0, 1))
    plt.savefig('theoretic_scores.png', dpi=144)
    plt.close()


if __name__ == '__main__':
    main()

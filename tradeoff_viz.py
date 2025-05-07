import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    p = np.logspace(0, 1.5, 15)
    x = np.linspace(0, 1, 100)

    xx, pp = np.meshgrid(x, p)
    zz = (1 - xx ** pp) ** (1 / pp)

    df = pd.DataFrame(dict(x=xx.ravel(), z=zz.ravel(), p=pp.ravel()))
    df['p'] = df['p'].astype(str)

    plt.figure(figsize=(6, 6))
    sns.set_style('darkgrid')
    sns.lineplot(data=df, x='x', y='z', hue='p')
    plt.gca().get_legend().remove()
    plt.xlabel('$ P(\\hat{y} = 1 | y = 1) $')
    plt.ylabel('$ P(\\hat{y} = 0 | y = 0) $')
    plt.savefig('tradeoff_viz.png', dpi=144)



if __name__ == '__main__':
    main()

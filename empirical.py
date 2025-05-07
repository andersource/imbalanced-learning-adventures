import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm


def main():
    res = []
    for class_sep in tqdm(np.linspace(.05, 3, 15).round(2), desc='Running stats...'):
        for positive_weight in np.logspace(0, 2.5, 15):
            for i in range(1000):
                X, y = make_classification(n_samples=10000, n_features=10, n_informative=5, n_redundant=0, weights=(.99,),
                                           n_clusters_per_class=2, class_sep=class_sep, flip_y=0)

                X_train, X_test, y_train, y_test = train_test_split(X, y)
                clf = DecisionTreeClassifier()
                sample_weight = np.where(y_train == 0, 1, positive_weight)
                clf.fit(X_train, y_train, sample_weight=sample_weight)

                res.append({
                    'f1': f1_score(y_test, clf.predict(X_test)),
                    'class_sep': class_sep,
                    'positive_weight': positive_weight
                })

    df = pd.DataFrame(res)
    df.to_csv('empirical_results.csv', index=False)


if __name__ == '__main__':
    main()

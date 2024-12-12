import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier

class ImportanceImputer():
    def __init__(self, similar_threshold=5, bin_numeric=False):
        self.similar_threshold = similar_threshold
        self.bin_numeric = bin_numeric

    def fit_transform(self, df, impute_feature):
        # Split categorical and numeric features
        numerical_features = list(set(df._get_numeric_data().columns) - set([impute_feature]))
        categorical_features = list((set(df.columns) - set(numerical_features)) - set([impute_feature]))
        # raise Exception

        if self.bin_numeric:
            # Use a decision tree classifier to bin numerical features according to the impute_feature
            for feat in numerical_features:
                # Step 1: Train a decision tree to predict colour from price
                paired_df = df.loc[df[impute_feature].notna()]

                X = paired_df[[feat]]  # numeric predictor
                y = paired_df[impute_feature]

                # You can tune max_depth to control how many splits (bins) you get
                tree = DecisionTreeClassifier(max_depth=2, random_state=42)
                tree.fit(X, y)

                # Step 2: Extract threshold cuts from the decision tree
                # Each non-leaf node uses a threshold that splits the data into two groups
                thresholds = []
                def traverse_tree(node=0):
                    # Check if node is split node
                    if tree.tree_.feature[node] != -2:
                        # -2 indicates a leaf node
                        threshold = tree.tree_.threshold[node]
                        thresholds.append(threshold)
                        traverse_tree(tree.tree_.children_left[node])
                        traverse_tree(tree.tree_.children_right[node])

                traverse_tree()
                thresholds = sorted(thresholds)

                # Step 3: Use these thresholds to bin price
                # For example, if thresholds = [5000, 15000],
                # you can create bins: [-inf, 5000), [5000, 15000), [15000, inf)
                bins = [-np.inf] + thresholds + [np.inf]
                labels = [f'Bin_{i}' for i in range(len(bins)-1)]

                categorical_features.append(feat+'_binned')

                df[feat+'_binned'] = pd.cut(df[feat], bins=bins, labels=labels)


        # Computer Cramer's V between each feature and the impute_feature
        cramer_dict = {}
        for feat in categorical_features:
            contingency_table = pd.crosstab(df[feat], df[impute_feature])

            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # number of rows and columns in the contingency table
            n = contingency_table.sum().sum()
            r, k = contingency_table.shape

            # Cramér's V
            cramer_v = np.sqrt((chi2 / n) / (min(r, k) - 1))
            
            cramer_dict[feat] = cramer_v


        cramer_dict = dict(sorted(cramer_dict.items(), key=lambda x:-x[1]))

        # Order features in descending order of cramer's V
        ordered_features = list(cramer_dict.keys())

        print("Matching features in order:", ordered_features)

        # Impute values where impute_feature is unknown
        missing_df = df.loc[df[impute_feature].isna()]

        def impute_value(row, full_df):
            subset = ordered_features[:]

            while subset:
                condition = True
                for feat in subset:
                    condition = condition & (full_df[feat] == row[feat])
                similar_entries = full_df.loc[condition & full_df[impute_feature].notna()]

                if len(similar_entries) >= self.similar_threshold:
                    return similar_entries[impute_feature].mode()[0]
                
                subset.pop()

            return full_df[impute_feature].mode()[0]

        print("Imputing", len(missing_df), "entries.")
        # Impute colours for the rows
        missing_df[impute_feature] = missing_df.apply(lambda r: impute_value(r, df), axis=1)

        if self.bin_numeric:
            # Drop binned feature columns
            for column in numerical_features:
                # missing_df.drop(columns=column+"_binned", inplace=True)
                df.drop(columns=column+"_binned", inplace=True)

        return pd.DataFrame(missing_df[impute_feature])
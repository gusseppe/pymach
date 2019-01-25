#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module will define the dataset. Thoughts:
    - Type of model: Classification, Regression, Clustering.
    - Data: save the dataset.
    - header: the dataset's header.
and so forth.
"""

# __all__ = [
#     'pipeline']

import numpy as np
import core.tools as tools

from textwrap import dedent


class Explain():
    """Define module.

    Parameters
    ------------
    data_name : string
    The dataset's name which is expected to be a csv file.

    header : list
    The dataset's header, i.e, the features and the class name.

    response : string
    The name of the variable will be used for prediction.

    Attributes
    -----------
    n_features : int
    number of features or predictors.

    samples : int
    Number of rows in the dataset.

    """


    def __init__(self, definer):
        """The init class.

        Parameters
        ----------
        typeModel : string
            String that indicates if the model will be trained for clasification
            or regression.
        className : string
            String that indicates which column in the dataset is the class.

        """
        self.problem_type = definer.problem_type
        self.num_features = definer.num_features
        self.cat_features = definer.cat_features
        # self.infer_algorithm = definer.infer_algorithm
        self.response = definer.response
        # self.data_path = definer.data_path
        self.data = definer.data
        self.explain = None

    def histogram(self):
        df_num = self.data[self.num_features]

        kurtosis_list = sorted(list(df_num.kurtosis().items()),
                               key=lambda x: abs(x[1]))
        explain = dedent(f'''
            ### Explanation: Histogram
            ###### **Most symmetrical variable**: {kurtosis_list[0][0]}
            ###### **Least symmetrical variable**: {kurtosis_list[-1][0]}
            ###### **Advice**: Apply log to {kurtosis_list[-1][0]}

        ''')
        self.explain = explain

        return explain

    def boxplot(self):
        df_num = self.data[self.num_features]

        Q1 = df_num.quantile(0.25)
        Q3 = df_num.quantile(0.75)
        IQR = Q3 - Q1
        temp_df = (df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))
        result = dict(temp_df.any().items())
        result = [k for k,v in result.items() if v]
        explain = dedent(f'''
                ### Explanation: Boxplot
                ###### **Features that contain outliers**: {result}
                ###### **Advice**: We are going to take into account this insight for you.
                ###### By the way, check if your dataset follows this pattern.

            ''')
        self.explain = explain

        return explain

    def correlation(self):
        df_num = self.data[self.num_features]

        corr = df_num.corr()
        threshold = 0.80
        # Select columns with correlations above threshold
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        explain = dedent(f'''
                ### Explanation: Correlation Matrix
                ###### **Features that are correlated (0.80 threshold)**: {to_drop}
                ###### **Advice**: Most likely, I'm going to drop a half of them to increase 
                ###### the overall performance. In the meantime, consider not include correlated variables 
                ###### the next time to model this dataset.
            ''')
        self.explain = explain

        return explain

    def scatter(self):
        explain = dedent(f'''
                    ### Explanation: Scatter
                    ###### **If it is a classification problem**:
                    ######  Check the clusters or groups between two features, are they overlapped?
                    ###### **If it is a regression problem**:
                    ######  Check how linear are the pair of features, are they non linear?
                ''')
        self.explain = explain

        return explain

    def parallel(self):
        explain = dedent(f'''
                    ### Explanation: Parallel Coordinates
                    ###### **If it is a classification problem**:
                    ######  Check the color and values belonging to each line, try to drag a feature!.
                ''')
        self.explain = explain

        return explain

    def explain_analyze(self, plot_type):

        if plot_type == 'hist':
            return self.histogram()
        elif plot_type == 'box':
            return self.boxplot()
        elif plot_type == 'corr':
            return self.correlation()
        elif plot_type == 'scatter':
            return self.scatter()
        elif plot_type == 'parallel':
            return self.parallel()

    def explain_model(self, plot_type):
        """Read the dataset.

        Returns
        -------
        out : ndarray
        # return dict_description
        """

        if plot_type == 'hist':
            kurtosis_list = sorted(list(self.data.kurtosis().items()), key=lambda x:abs(x[1]))
            hist_explain = dedent(f'''
                ## Explanation
                ###### **Most symmetrical variable**: {kurtosis_list[0][0]}
                ###### **Least symmetrical variable**: {kurtosis_list[-1][0]}
                ###### **Advice**: Apply log to {kurtosis_list[-1][0]}

            ''')
            return hist_explain

    # def save(self, path):
    #     with open(path, "w") as file:
    #         file.write(self.explain)

    def save(self, path):
        # hickle.dump(self.explain, path, mode='w')
        # tools.plotlyfig2json(self.explain, path)
        with open(path, "w") as file:
            file.write(self.explain)


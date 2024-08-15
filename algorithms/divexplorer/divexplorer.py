import numpy as np
import pandas as pd

from algorithms.divexplorer.utils_FPgrowth import fpgrowth_cm


def get_df_one_hot_encoding(df_input):
    # One-hot encoding of the dataframe
    attributes = df_input.columns
    X_one_hot = pd.get_dummies(df_input, prefix_sep="=", columns=attributes)
    X_one_hot.reset_index(drop=True, inplace=True)
    return X_one_hot

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))





def get_t_value_bayesian(positives, negatives, index_all_data):
    pos_plus_one = 1 + positives
    neg_plus_1 = 1 + negatives

    # Mean beta distribution
    mean_beta = pos_plus_one / (pos_plus_one + neg_plus_1)

    # Variance beta distribution
    variance_beta = (pos_plus_one * neg_plus_1) / (
        (pos_plus_one + neg_plus_1) ** 2 * (pos_plus_one + neg_plus_1 + 1)
    )

    mean_dataset, var_dataset = mean_beta[index_all_data], variance_beta[index_all_data]
    t_test_value = (abs(mean_beta - mean_dataset)) / (
        (variance_beta + var_dataset) ** 0.5
    )

    return t_test_value


def compute_welch_t_test(
    squared_values, support_count_values, mean_values, squared_values_p, support_count_values_p, mean_values_p
):
    variance_values = squared_values / support_count_values - mean_values**2
    variance_values_p = squared_values_p / support_count_values_p - mean_values_p**2

    # Welch's t-test
    t_test_welch = (abs(mean_values - mean_values_p)) / (
        (
            variance_values / support_count_values
            + variance_values_p / support_count_values_p
        )
        ** 0.5
    )
    return t_test_welch


class DivergenceExplorer:
    def __init__(
        self,
        df_discrete,
        is_one_hot_encoding=False,
    ):
        # df_discrete: pandas dataframe with discrete values
        self.df_discrete = df_discrete

        # is_one_hot_encoding: boolean, if True, the dataframe attributes are already one-hot encoded
        self.is_one_hot_encoding = is_one_hot_encoding

    def get_pattern_divergence(
        self,
        min_support: float,
        boolean_outcomes: list = None,
        quantitative_outcomes: list = None,
        attributes: list = None,
        FPM_algorithm="fpgrowth",
        show_coincise=True,
        group_1_column="",
        group_2_column=""
    ):
        """
        Args:
            min_support (float): minimum support value for the pattern
            boolean_outcomes (list): list of boolean outcomes
            quantitative_outcomes (list): list of quantitative outcomes
            attributes (list): list of attributes to consider
            FPM_algorithm (str): algorithm to use for frequent pattern mining
            show_coincise (bool): if True, the output is more concise, returning only the average, the divergence and the t value
        """

        assert FPM_algorithm in [
            "fpgrowth",
            "apriori",
        ], f"{FPM_algorithm} algorithm is not handled. Qe integrate the DivExplorer computation in 'fpgrowth' and 'apriori' algorithms."

        # Sets the default values for lists. 
        quantitative_outcomes = quantitative_outcomes or []
        boolean_outcomes = boolean_outcomes or []
        attributes = attributes or []

        assert (
            len(boolean_outcomes) > 0 or len(quantitative_outcomes) > 0
        ), "At least one outcome must be specified."

        assert (
            len(boolean_outcomes) == 0 or len(quantitative_outcomes) == 0
        ), "Only one type of outcome must be specified."

        if attributes == []:
            # Get all attributes except outcomes
            attributes = [
                attr
                for attr in list(self.df_discrete.columns)
                if attr not in boolean_outcomes + quantitative_outcomes + [group_2_column,group_1_column]
            ]

        # Get only the attributes specified
        df_discrete = self.df_discrete[attributes]

        len_dataset = len(self.df_discrete)

        # If it is not already one-hot encoded, we one-hot encode it
        df_ohe = get_df_one_hot_encoding(df_discrete)

        # If there are quantitative outcomes, we compute the squared outcome
        df_outcomes = pd.DataFrame()
        df_outcomes["OUTCOME"] = self.df_discrete[quantitative_outcomes[0]].copy()
        df_outcomes["GROUP_1"] = self.df_discrete[group_1_column].copy()
        df_outcomes[f"GROUP_1_outcome"] = df_outcomes["GROUP_1"] * df_outcomes["OUTCOME"]
        df_outcomes.loc[:, f"GROUP_2"] = (
            np.array(self.df_discrete[group_2_column].values)
        )
        df_outcomes[f"GROUP_2_outcome"] = df_outcomes["GROUP_2"] * df_outcomes["OUTCOME"]

        if FPM_algorithm == "fpgrowth":
            df_divergence = fpgrowth_cm(
                df_ohe.copy(),  # Df with one-hot encoded attributes
                df_outcomes,  # Df with outcomes
                min_support=min_support,  # Minimum support
                columns_accumulate=list(df_outcomes.columns),  # Columns to accumulate
            )
        else:
            # We use the apriori algorithm
            if quantitative_outcomes:
                raise ValueError(
                    "The apriori implementation is available only for boolean outcomes."
                )
            df_with_outcomes = pd.concat([df_ohe, df_outcomes], axis=1)

            from algorithms.divexplorer.utils_apriori import apriori_divergence

            df_divergence = apriori_divergence(
                df_ohe.copy(),
                df_with_outcomes,
                min_support=min_support,
            )

        all_dataset_row = {"support": 1, "itemset": frozenset()}

        cols_to_drop = []
        squared_cols_to_drop = []
        
        if boolean_outcomes:
            for boolean_outcome in boolean_outcomes:
                # The result is average when non considering the bottom values

                def compute_outcome(positive_outcomes, negative_outcomes):
                    return positive_outcomes / (positive_outcomes + negative_outcomes)

                positive_col_name = f"{boolean_outcome}_positive"
                negative_col_name = f"{boolean_outcome}_negative"
                bottom_col_name = f"{boolean_outcome}_bottom"

                df_divergence[boolean_outcome] = compute_outcome(
                    df_divergence[positive_col_name],
                    df_divergence[negative_col_name],
                )

                cols_to_drop.extend(
                    [positive_col_name, negative_col_name, bottom_col_name]
                )

                # Add the info of the all dataset row
                for column_name in [
                    positive_col_name,
                    negative_col_name,
                    bottom_col_name,
                ]:
                    all_dataset_row[column_name] = df_outcomes[column_name].sum()

                all_dataset_row[
                    f"{boolean_outcome}_div"
                ] = 0  # The divergence is 0 by definition
                all_dataset_row[
                    f"{boolean_outcome}_t"
                ] = 0  # The t value is 0 by definition

                # Compute the average of the all dataset row -- as above
                overall_average = compute_outcome(
                    all_dataset_row[positive_col_name],
                    all_dataset_row[negative_col_name],
                )

                # Add the average to the all dataset row
                all_dataset_row[boolean_outcome] = overall_average

                # Compute the divergence
                df_divergence[f"{boolean_outcome}_div"] = (
                    df_divergence[boolean_outcome] - overall_average
                )

                # Compute the t value

                # Get the positive and negative values
                pos, neg = (
                    df_divergence[positive_col_name].values,
                    df_divergence[negative_col_name].values,
                )

                # We append the pos and neg values of the all dataset row
                pos = np.concatenate([[all_dataset_row[positive_col_name]], pos])
                neg = np.concatenate([[all_dataset_row[negative_col_name]], neg])
                t = get_t_value_bayesian(pos, neg, 0)
                df_divergence[f"{boolean_outcome}_t"] = t[
                    1:
                ]  # We omit the all dataset row

        else:
            for quantitative_outcome in quantitative_outcomes:
                df_divergence[quantitative_outcome] = (
                    df_divergence["OUTCOME"]
                    / (df_divergence["support"] * len_dataset).round()
                )

                df_divergence[f"{quantitative_outcome}_group1"] = (df_divergence[f"GROUP_1_outcome"]
                                                                      / (df_divergence[f"GROUP_1"])
                                                                      .round())
                df_divergence[f"{quantitative_outcome}_group2"] = (df_divergence[f"GROUP_2_outcome"]
                                                                   / (df_divergence[f"GROUP_2"])
                                                                   .round())

                # Compute the divergence
                df_divergence[f"{quantitative_outcome}_div"] = (
                    df_divergence[f"{quantitative_outcome}_group1"] - df_divergence[f"{quantitative_outcome}_group2"]
                )

        df_divergence["length"] = df_divergence["itemset"].str.len()
        df_divergence["support_count"] = (
            df_divergence["support"] * len_dataset
        ).round()
        df_divergence = df_divergence.reset_index(drop=True)
        df_divergence = df_divergence.drop(["OUTCOME", "GROUP_1_outcome", "GROUP_2_outcome"], axis=1)

        df_divergence.sort_values("support", ascending=False, inplace=True)
        df_divergence = df_divergence.reset_index(drop=True)

        if show_coincise:
            df_divergence = df_divergence.drop(columns=cols_to_drop)
        
        return df_divergence

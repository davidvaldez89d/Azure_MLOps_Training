import argparse
import os

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()
    # Add mlflow logging to this script

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    credit_df = pd.read_excel(args.data, header=1, index_col=0)

    # YOUR CODE HERE

    # Log the number of samples as "num_samples"
    mlflow.log_metric("num_samples", credit_df.shape[0])
    # Log the number of features as "num_features"
    mlflow.log_metric("num_features", credit_df.shape[1])

    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )

    credit_train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    credit_test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # stop mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()

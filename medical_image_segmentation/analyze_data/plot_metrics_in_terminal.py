import plotext as plt
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, help="Which headers to plot")
    parser.add_argument("--filenames", nargs='+', help="One or more CSV files to process")

    return parser.parse_args()

def main():
    args = parse_args()

    combined_df = pd.DataFrame()
    for filename in args.filenames:
        df = pd.read_csv(filename)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.sort_values(by="epoch", inplace=True)

    for metric in args.metrics:
        if metric in combined_df.columns:
            data = combined_df[metric].dropna()
            plt.plot(data, label=metric)
        else:
            print(f"Warning: Metric '{metric}' not found in the CSV files.")

    plt.title("Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

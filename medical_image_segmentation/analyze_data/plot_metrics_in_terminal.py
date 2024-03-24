import plotext as plt
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", nargs="+", required=True, help="Which headers to plot"
    )
    parser.add_argument("--files", nargs="+", help="One or more CSV files to process")
    parser.add_argument(
        "--ymin", type=float, required=False, help="Minimum y value to plot"
    )
    parser.add_argument(
        "--ymax", type=float, required=False, help="Maximum y value to plot"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    combined_df = pd.DataFrame()
    for filename in args.files:
        df = pd.read_csv(filename)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.sort_values(by="epoch", inplace=True)

    for metric in args.metrics:
        if metric in combined_df.columns:
            data = combined_df[metric].dropna()
            plt.plot(data, label=metric)
        else:
            print(f"Warning: Metric '{metric}' not found in the CSV files.")

    plt.title("Metrics over Time")
    plt.xlabel("Time")
    plt.ylabel("Metric Value")

    # Set the y-axis limits if specified
    if args.ymin is not None or args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)

    plt.show()


if __name__ == "__main__":
    main()

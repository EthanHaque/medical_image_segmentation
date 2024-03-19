import plotext as plt
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+', help="One or more CSV files to process")
    
    return parser.parse_args()


def main():
    args = parse_args()

    combined_df = pd.DataFrame()
    for filename in args.filenames:
        df = pd.read_csv(filename)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.sort_values(by="epoch", inplace=True)

    losses = combined_df["train_loss_epoch"].dropna()
    plt.plot(losses)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 2)
    plt.show()

if __name__ == "__main__":
    main()


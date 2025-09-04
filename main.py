import argparse
from train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train an LSTM model on any CSV file.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file")
    parser.add_argument("--seperator", type=str, default=",",
                        help="(--seperator) Seperated Values")
    parser.add_argument("--column", type=str, required=True,
                        help="Target column to predict")
    parser.add_argument("--window", type=int, default=10,
                        help="Sequence length for LSTM")
    parser.add_argument("--val_size", type=int, default=30,
                        help="Validation set size")
    parser.add_argument("--test_size", type=int,
                        default=30, help="Test set size")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=2,
                        help="Hidden layer size")

    args = parser.parse_args()

    model = train_model(
        csv_path=args.csv,
        seperator=args.seperator,
        target_column=args.column,
        window=args.window,
        val_size=args.val_size,
        test_size=args.test_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden
    )
    print("✅ Training complete and model ready.")


if __name__ == "__main__":
    main()

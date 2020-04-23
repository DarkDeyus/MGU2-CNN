import os


def main():
    datadir = "data"
    accfile = "accuracy"
    for dir in os.listdir(datadir):
        with open(os.path.join(datadir, dir, accfile), 'r') as f:
            print(f"Accuracy of {dir} - {f.read()}")


if __name__ == "__main__":
    main()
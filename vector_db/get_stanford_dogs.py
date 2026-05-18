from pathlib import Path

from datasets import load_dataset, load_from_disk

from consts import DATASET_FOLDER, TEST_FOLDER


def check_for_local_dataset():
    """Checks the current directory for the dataset folder.

    Returns True if the '.stanford_dogs' folder is found.
    """
    curr_dir = Path(__file__).parent.absolute()
    is_dataset_locally = False
    folders = [path.name for path in curr_dir.iterdir() if path.is_dir()]

    if DATASET_FOLDER in folders:
        is_dataset_locally = True

    return is_dataset_locally


def get_stanford_dogs():
    """Downloads the dataset from Hugging Face and saves it locally."""
    train = load_dataset("maurice-fp/stanford-dogs", split="train")

    train.save_to_disk(DATASET_FOLDER)

    test = load_dataset("maurice-fp/stanford-dogs", split="test")
    test.save_to_disk(TEST_FOLDER)


if __name__ == "__main__":
    print("checking for dataset locally...")
    if not check_for_local_dataset():
        print("downloading dataset...")
        get_stanford_dogs()

    try:
        dataset = load_from_disk(DATASET_FOLDER)
        print("Dataset loaded successfully!")

        print(f"Total images found: {len(dataset)}")
        print(f"Features in dataset: {dataset.features.keys()}")

    except Exception as e:
        print(f"could not load dataset: {e}")

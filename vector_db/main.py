from pathlib import Path

from get_clip import get_clip_model


def check_for_local_clip():
    """Checks the current directory for the CLIP model.

    Returns True if found ".model_clip" folder
    """
    curr_dir = Path(__file__).parent.absolute()
    is_model_locally = False
    folders = [path.name for path in curr_dir.iterdir() if path.is_dir()]
    if ".clip_model" in folders:
        is_model_locally = True

    return is_model_locally


if __name__ == "__main__":
    print("Checking for CLIP locally...")
    if not check_for_local_clip():
        get_clip_model()

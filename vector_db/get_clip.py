from pathlib import Path

from transformers import AutoModel, AutoProcessor

from consts import CLIP_FOLDER


def check_for_local_clip():
    """Checks the current directory for the CLIP model.

    Returns True if found CLIP_FOLDER folder
    """
    curr_dir = Path(__file__).parent.absolute()
    is_model_locally = False
    folders = [path.name for path in curr_dir.iterdir() if path.is_dir()]
    if CLIP_FOLDER in folders:
        is_model_locally = True

    return is_model_locally


def get_clip_model():
    model_name = "openai/clip-vit-base-patch32"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    processor.save_pretrained(CLIP_FOLDER)
    model.save_pretrained(CLIP_FOLDER)

    print("Model and processor saved locally")


if __name__ == "__main__":
    get_clip_model()

from transformers import AutoModel, AutoProcessor


def get_clip_model():
    model_name = "openai/clip-vit-base-patch32"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    processor.save_pretrained("./.clip_model")
    model.save_pretrained("./.clip_model")

    print("Model and processor saved locally")


if __name__ == "__main__":
    get_clip_model()

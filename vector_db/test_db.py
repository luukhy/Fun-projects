import torch
from datasets import load_from_disk
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoProcessor

from consts import CLIP_FOLDER, QDRANT_COLLECTION_NAME, QDRANT_LOCATION, TEST_FOLDER


def test_search():
    test = load_from_disk(TEST_FOLDER)

    test_item = test[0]

    image = test_item["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    true_breed = test.features["label"].int2str(test_item["label"])

    processor = AutoProcessor.from_pretrained(CLIP_FOLDER)
    model = AutoModel.from_pretrained(CLIP_FOLDER)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    result = outputs.pooler_output.tolist()[0]

    qdrant = QdrantClient(QDRANT_LOCATION)

    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=result,
        limit=5,
    )

    print(f"True breed: {true_breed}")
    for i, hit in enumerate(hits.points):
        breed = hit.payload["breed"]
        score = hit.score
        print(f"{i + 1}. Breed: {breed.ljust(25)} | Score: {score:.4f}")


if __name__ == "__main__":
    test_search()

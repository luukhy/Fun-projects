import hashlib
import io
import uuid

import torch
from datasets import load_from_disk
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from consts import (
    CLIP_FOLDER,
    DATASET_FOLDER,
    FEATURE_SIZE,
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCATION,
)
from get_clip import check_for_local_clip, get_clip_model
from get_stanford_dogs import check_for_local_dataset, get_stanford_dogs


def clip_init():
    processor = None
    model = None
    print("checking for CLIP locally...")
    if not check_for_local_clip():
        print("downloading clip...")
        get_clip_model()

    try:
        processor = AutoProcessor.from_pretrained(CLIP_FOLDER)
        model = AutoModel.from_pretrained(CLIP_FOLDER)
        print("CLIP loaded")
    except Exception as e:
        print(f"could not load CLIP: {e}")
    return processor, model


def dataset_init():
    dataset = None
    print("Checking for dataset locally...")
    if not check_for_local_dataset():
        print("Downloading dataset...")
        get_stanford_dogs()

    try:
        dataset = load_from_disk(DATASET_FOLDER)
        print("Dataset loaded successfully!")

        print(f"Total images found: {len(dataset)}")
        print(f"Features in dataset: {dataset.features.keys()}")

    except Exception as e:
        print(f"Could not load dataset: {e}")

    return dataset


def qdrant_init():
    qdrant = QdrantClient(QDRANT_LOCATION)
    if not qdrant.collection_exists(QDRANT_COLLECTION_NAME):
        print(f"Creating Qdrant collection: {QDRANT_COLLECTION_NAME}...")
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=FEATURE_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {QDRANT_COLLECTION_NAME} found")
    return qdrant


def main():
    processor, model = clip_init()
    if processor is None or model is None:
        return

    dataset = dataset_init()
    if dataset is None:
        return

    qdrant = qdrant_init()

    batch_size = 32
    points = []

    print("Extracting embeddings and uploading...")
    for item in tqdm(dataset):
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        label_int = item["label"]
        breed = dataset.features["label"].int2str(label_int)

        image_byte_arr = io.BytesIO()
        image.save(image_byte_arr, format="JPEG")
        img_bytes = image_byte_arr.getvalue()

        hasher = hashlib.md5(img_bytes)  # 32 charachter string
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, hasher.hexdigest())
        )  # for Qdrant requirements must be uuid

        already_exists = qdrant.retrieve(
            collection_name=QDRANT_COLLECTION_NAME, ids=[point_id]
        )  # not a bool, returns a list of retrieved items if it finds them
        if already_exists:
            continue

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        vector = outputs.pooler_output.tolist()[0]

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={"breed": breed, "dataset_label": label_int},
        )
        points.append(point)

        if len(points) >= batch_size:
            qdrant.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
            points = []

    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    print("Succesfully populated Qdrant")


if __name__ == "__main__":
    main()

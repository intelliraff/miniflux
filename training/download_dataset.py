from datasets import load_dataset

dataset = load_dataset(
    "Pin-ky/coco-blip-captions"
)

print(dataset)

dataset.save_to_disk(
    "../data/coco"
)

print("Dataset saved!")
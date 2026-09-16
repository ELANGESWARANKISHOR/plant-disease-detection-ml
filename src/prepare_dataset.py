import os
import shutil
import random

# Original dataset location
SOURCE_DIR = "data/plant_village/potatoes"

# New dataset location
OUTPUT_DIR = "data/dataset"

# Classes we want to use
CLASSES = [
    "early_blight",
    "healthy",
    "late_blight"
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Make results reproducible
random.seed(42)


def create_directories():
    for split in ["train", "validation", "test"]:
        for class_name in CLASSES:
            directory = os.path.join(
                OUTPUT_DIR,
                split,
                class_name
            )

            os.makedirs(directory, exist_ok=True)


def split_images():
    for class_name in CLASSES:

        source = os.path.join(
            SOURCE_DIR,
            class_name,
            "color",
            "images"
        )

        images = [
            file for file in os.listdir(source)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(f"\n{class_name}")
        print(f"Total: {total}")
        print(f"Training: {len(train_images)}")
        print(f"Validation: {len(val_images)}")
        print(f"Testing: {len(test_images)}")

        copy_images(train_images, source, "train", class_name)
        copy_images(val_images, source, "validation", class_name)
        copy_images(test_images, source, "test", class_name)


def copy_images(images, source, split, class_name):

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        class_name
    )

    for image in images:

        source_path = os.path.join(source, image)
        destination_path = os.path.join(destination, image)

        shutil.copy2(source_path, destination_path)


if __name__ == "__main__":

    print("Creating dataset directories...")

    create_directories()

    print("Splitting images...")

    split_images()

    print("\nDataset preparation complete!")
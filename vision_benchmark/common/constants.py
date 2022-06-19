from vision_datasets import DatasetHub
import pathlib

VISION_DATASET_STORAGE = 'https://irisdatasets.blob.core.windows.net/share'


def get_dataset_hub():
    vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / 'resources' / 'datasets' / 'vision_datasets.json').read_text()
    hub = DatasetHub(vision_dataset_json)

    return hub

from vision_datasets import DatasetHub
import pathlib

VISION_DATASET_STORAGE = 'https://cvinthewildeus.blob.core.windows.net/datasets?sp=r&st=2023-08-28T01:41:20Z&se=3023-08-28T09:41:20Z&sv=2022-11-02&sr=c&sig=Msoq5dIl%2Fve6F01edGr8jgcZUt7rtsuJ896xvstSNfM%3D'


def get_dataset_hub():
    vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / 'resources' / 'datasets' / 'vision_datasets.json').read_text()
    hub = DatasetHub(vision_dataset_json)

    return hub

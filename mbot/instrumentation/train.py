from mbot.modeling.datasets.facebook_dataset import FacebookDataset
from mbot.modeling.datasets.cornell_dataset import CornellDataset
from mbot.modeling.datasets.combined_dataset import CombinedDataset
from mbot.modeling.models.preprocessor import Preprocessor
from mbot.modeling.models.seq2seq import Seq2Seq

__author__ = 'Elisha Yadgaran'

DATASETS = [FacebookDataset, CornellDataset]


def create_dataset(datasets, load_raw_data=False):
    def get_dataset_class(name):
        for i in DATASETS:
            if i.__name__ == name:
                return i

    dataset_classes = [get_dataset_class(i) for i in datasets]
    datasets = [dataset(load_raw_data) for dataset in dataset_classes]
    dataset = CombinedDataset(datasets)

    return dataset


def preprocess_dataset(dataset):
    preprocessor = Preprocessor()
    preprocessor.fit(dataset)
    preprocessor.save()

    return preprocessor


def create_model(preprocessor):
    model = Seq2Seq(preprocessor)
    model.compile_model()

    return model


if __name__ == '__main__':
    dataset = create_dataset(['FacebookDataset', 'CornellDataset'])
    preprocessor = preprocess_dataset(dataset)
    model = create_model(preprocessor)
    model.train(dataset.x, dataset.y)
    model.save()

from azureml.core import Dataset
from azureml.data.datapath import DataPath

from utility import connection_azure

import dotenv
import random
import shutil
import math
import glob
import cv2
import os

dotenv.load_dotenv()
emotions = os.environ.get('emotions').split(',')
split_factor = float(os.environ.get('split_factor'))

def process_upload(ws, datasets, folder, emotion):
    path_raw = os.path.join(folder, 'raw', emotion)
    path_processed = os.path.join(folder, 'processed', emotion)

    datasets[emotion].download(path_raw, overwrite=True)
    path_images = glob.glob(f'{path_raw}/*.png')

    for path_image in path_images:
        image = cv2.imread(path_image)
        image = cv2.resize(image, (48, 48))
        cv2.imwrite(os.path.join(path_processed, path_image.split('/')[-1]), image)

    processed_dataset = Dataset.File.upload_directory(
        src_dir=os.path.join(path_processed),
        target=DataPath(
            datastore=ws.get_default_datastore(),
            path_on_datastore=f'processed/{emotion}'
        ),
        overwrite=True,
        show_progress=False
    )

    registered_dataset = processed_dataset.register(
        ws,
        name=f'processed_{emotion}',
        description=f'processed {emotion} dataset',
        tags={'emotions': emotion, 'git-sha': os.environ.get('git_sha')},
        create_new_version=True
    )

    print(f'Dataset id: {registered_dataset.id}')

    shutil.rmtree(path_raw)
    shutil.rmtree(path_processed)

def prepare(ws):
    datasets = Dataset.get_all(workspace=ws)
    folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(folder, exist_ok=True)

    for emotion in emotions:
        os.makedirs(os.path.join(folder, 'raw', emotion), exist_ok=True)
        os.makedirs(os.path.join(folder, 'processed', emotion), exist_ok=True)
        process_upload(ws, datasets, folder, emotion)

def split(ws):
    datapath_training = []
    datapath_testing = []
    default_datastore = ws.get_default_datastore()

    for emotion in emotions:
        dataset = Dataset.get_by_name(ws, f'processed_{emotion}')
        images = [img for img in dataset.to_path() if img.split('.')[-1] == 'png']
        images = [(default_datastore, f'processed/{emotion}{path}') for path in images]

        random.seed(42)
        random.shuffle(images)

        images_percentage = math.ceil(len(images) * split_factor)
        images_training = images[images_percentage:]
        images_testing = images[:images_percentage]

        datapath_training.extend(images_training)
        datapath_testing.extend(images_testing)

    dataset_training = Dataset.File.from_files(path=datapath_training)
    dataset_testing = Dataset.File.from_files(path=datapath_testing)

    dataset_training = dataset_training.register(
        ws,
        name = os.environ.get('dataset_training'),
        description = f'Emotional images to train',
        tags = {'AI_Model': 'CNN', 'Split': os.environ.get('split_factor'), 'Type': 'training'},
        create_new_version = True
    )

    dataset_testing = dataset_testing.register(
        ws,
        name = os.environ.get('dataset_testing'),
        description = f'Emotional images to test',
        tags = {'AI_Model': 'CNN', 'Split': os.environ.get('split_factor'), 'Type': 'testing'},
        create_new_version = True
    )

if __name__ == '__main__':
    print('Preparation')
    ws = connection_azure()
    print('Connected')

    pipeline_process = os.environ.get('pipeline_process')
    pipeline_split = os.environ.get('pipeline_split')

    if pipeline_process == 'true':
        print('Processing')
        prepare(ws)

    if pipeline_split == 'true':
        print('Splitting')
        split(ws)
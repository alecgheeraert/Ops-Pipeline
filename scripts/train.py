from tensorflow import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from azureml.core import Run

from utility import *
import argparse
import random
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--folder-training', type=str, dest='folder_training', help='training folder mounting point.')
parser.add_argument('--folder-testing', type=str, dest='folder_testing', help='testing folder mounting point.')
parser.add_argument('--max-epochs', type=int, dest='max_epochs', help='The maximum epochs to train.')
parser.add_argument('--learning-rate-init', type=float, dest='learning_rate_init', help='The initial learning rate to use.')
parser.add_argument('--batch-size', type=int, dest='batch_size', help='The batch size to use during training.')
parser.add_argument('--patience', type=int, dest='patience', help='The patience for the Early Stopping.')
parser.add_argument('--model-name', type=str, dest='model_name', help='The name of the model to use.')
args = parser.parse_args()

folder_training = args.folder_training
folder_testing = args.folder_testing
max_epochs = args.max_epochs
learning_rate_init = args.learning_rate_init
batch_size = args.batch_size
patience = args.patience
model_name = args.model_name

print('Training folder:', folder_training)
print('Testing folder:', folder_testing)


print('Paths:')
paths_training = glob.glob(os.path.join('/data/train', '**', '*.png'), recursive=True)
paths_testing = glob.glob(os.path.join('/data/test', '**', '*.png'), recursive=True)

print("Training samples:", len(paths_training))
print("Testing samples:", len(paths_testing))

random.seed(42)
random.shuffle(paths_training)

print(paths_training[:3])
print(paths_testing[:3])

X_train = getFeatures(paths_training)
y_train = getTargets(paths_training)

X_test = getFeatures(paths_testing)
y_test = getTargets(paths_testing)

print('Shapes:')
print(X_train.shape)
print(X_test.shape)

LABELS, y_train, y_test = encodeLabels(y_train, y_test)
print('One Hot Shapes:')

print(y_train.shape)
print(y_test.shape)

model_path = os.path.join('outputs', model_name)
os.makedirs(model_path, exist_ok=True)
run = Run.get_context()


cb_save_best_model = keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

cb_early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=patience,
    verbose=1,
    restore_best_weights=True
)

cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    factor=.5, 
    patience=4, 
    verbose=1
)

class LogToAzure(keras.callbacks.Callback):
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)


opt = SGD(learning_rate=learning_rate_init)
model = buildModel((48, 48, 3), 4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

aug = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode="nearest"
)

history = model.fit( 
    aug.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=max_epochs,
    callbacks=[
        LogToAzure(run),
        cb_save_best_model, 
        cb_early_stop, 
        cb_reduce_lr_on_plateau
    ] 
)

print("Evaluating network...")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=['happy', 'sad', 'fearful', 'angry']))

cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(cf_matrix)

cmtx = {
    "schema_type": "confusion_matrix",
    # "parameters": params,
    "data": {
        "class_labels": ['happy', 'sad', 'angry', 'fearful'],
        "matrix": [[int(y) for y in x] for x in cf_matrix]
    }
}

run.log_confusion_matrix('Confusion matrix - error rate', cmtx)
np.save('outputs/confusion_matrix.npy', cf_matrix)
print('Training Done')
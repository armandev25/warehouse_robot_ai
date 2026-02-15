import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

IMG_SIZE = (224,224)
BATCH = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "data/dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="training"
)

val = datagen.flow_from_directory(
    "data/dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="validation"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
out = Dense(train.num_classes, activation="softmax")(x)
model = Model(base.input, out)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train, validation_data=val, epochs=5)

model.save("models/warehouse_classifier.keras")

# ---- evaluation ----
preds = model.predict(val)
y_pred = np.argmax(preds, axis=1)
y_true = val.classes

print(classification_report(y_true, y_pred, target_names=list(val.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=val.class_indices.keys(),
            yticklabels=val.class_indices.keys())
plt.show()

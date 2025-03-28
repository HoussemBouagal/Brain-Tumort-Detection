import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Chemin du dataset
dataset_path = "E:\\Brain Tumor MRI Dataset\\Train"
test_dataset_path = "E:\\Brain Tumor MRI Dataset\\Test"

# Vérification des chemins
def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Chemin introuvable: {path}. Assurez-vous que le dataset est extrait!")
check_path(dataset_path)
check_path(test_dataset_path)

# Paramètres
taille_image = (224, 224)
taille_batch = 32
taux_apprentissage = 0.0001

# Générateur de données avec augmentation
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% des données pour validation
)

# Chargement des données d'entraînement et validation
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=taille_image,
    batch_size=taille_batch,
    class_mode='categorical',
    subset="training"
)
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=taille_image,
    batch_size=taille_batch,
    class_mode='categorical',
    subset="validation"
)

# Chargement des données de test
test_data = datagen.flow_from_directory(
    test_dataset_path,
    target_size=taille_image,
    batch_size=taille_batch,
    class_mode='categorical',
    shuffle=False
)

# Récupération des classes
class_names = list(train_data.class_indices.keys())
print(f"📌 Classes disponibles: {class_names}")

# Modèle MobileNetV2 (Transfer Learning)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # On gèle les couches pré-entraînées

# Construction du modèle
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

# Compilation du modèle
model.compile(
    optimizer=Adam(learning_rate=taux_apprentissage),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Affichage du résumé du modèle
model.summary()

# Callbacks pour l'optimisation
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraînement du modèle
epochs = 25
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[lr_scheduler, early_stopping]
)

# Sauvegarde du modèle
model_path = os.path.join(os.getcwd(), "brain_tumor_model.keras")
model.save(model_path)
print(f"✅ Modèle sauvegardé avec succès à: {model_path}")

# Fonction de visualisation
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Épochs')
    plt.ylabel('Précision')
    plt.legend()
    plt.title('📊 Progression de la précision')

    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entraînement')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Épochs')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('📉 Progression de la perte')

    plt.show()

plot_metrics(history)

# Évaluation sur les données de test
test_loss, test_acc = model.evaluate(test_data)
print(f"📊 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
# Affichage de l'accuracy finale après entraînement
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"✅ Accuracy finale - Entraînement: {final_train_acc:.4f}, Validation: {final_val_acc:.4f}")
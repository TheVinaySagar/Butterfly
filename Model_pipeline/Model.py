from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
import sys
import os.path
from tensorflow.keras.layers import ( # type: ignore 
    Dense,
    Input,
    Dropout,
    Flatten,
    RandomFlip,
    RandomRotation,
)
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.optimizers import RMSprop # type: ignore
# Add the parent directory to the system path for module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
def get_model_classification(
        input_shape=(224,224, 3),
        model="VGG16",
        weights="imagenet",
        n_classes=75,
        multi_class=False,
):
    input = Input(input_shape)
    
    if model == "VGG16":
        base_model = VGG16(
            include_top=False, input_shape=input_shape, weights=weights  # Fixed typo here
        )
        
    set_trainable = False

    # Freeze layers until 'block5_conv1'
    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        layer.trainable = set_trainable  # Simplified this line
    
    # Define data augmentation
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
    ])
    
    # Build the model
    x = data_augmentation(input)
    x = base_model(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(n_classes, activation='softmax')(x)  # Use n_classes parameter
    
    model = Model(inputs=[input], outputs=output)
    model.compile(
        optimizer=RMSprop(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()  # Optional: Keep for debugging or exploration
    return model

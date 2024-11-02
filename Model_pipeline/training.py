import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (ModelCheckpoint,ReduceLROnPlateau, EarlyStopping) # type: ignore
import sys
import os.path

# Add the parent directory to the system path for module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from Model_pipeline.Model import get_model_classification
from Model_pipeline.training_utilities import (
    create_data_generators
)


def train_from_csv(csv_path, data_config_path, training_config_path):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    with open(data_config_path, "r") as f:
        data_config = yaml.load(f, yaml.SafeLoader)
    with open(training_config_path, "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    train_gen,val_gen = create_data_generators(train_df,val_df,data_config["images_base_path"],tuple(data_config["target_size"]),training_config["batch_size"])

    model = get_model_classification(
        input_shape=tuple(data_config["input_shape"]),
        n_classes=75,
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=3,
        restore_best_weights=True 
    )

    history = model.fit(
    train_gen,
    epochs=training_config["epochs"],
    validation_data=val_gen,
    callbacks=[early_stopping]
)


if __name__ == "__main__":
    """
    python train.py --csv_path "../example/data.csv" \
                    --data_config_path "../example/data_config.yaml" \
                    --training_config_path "../example/training_config.yaml"
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="csv_path", default="../example/data.csv")
    parser.add_argument(
        "--data_config_path",
        help="data_config_path",
        default="../example/data_config.yaml",
    )
    parser.add_argument(
        "--training_config_path",
        help="training_config_path",
        default="../example/training_config.yaml",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    data_config_path = args.data_config_path
    training_config_path = args.training_config_path

    train_from_csv(
        csv_path=csv_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
    )
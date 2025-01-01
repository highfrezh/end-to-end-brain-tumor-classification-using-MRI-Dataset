import os
import pickle
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json



class Evaluation:
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config

    def load_saved_data_pickle(self):
        """
        Load the saved validation datasets.
        """
        with open(os.path.join(self.config.val_data_path, "val_data.pkl"), "rb") as f:
            X_val, Y_val = pickle.load(f)

            print(X_val.shape)
            print(Y_val.shape)

        return X_val, Y_val

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        print(f"Loading model from {path}")
        return tf.keras.models.load_model(path)
    

    def evaluation(self,X_val, Y_val):
        model = self.load_model(self.config.path_of_model)
        print("Model loaded successfully")
        self.score = model.evaluate(X_val, Y_val)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
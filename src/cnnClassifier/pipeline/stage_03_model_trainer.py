from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training, DataPreprocessor
from cnnClassifier import logger




STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        prepare_base_model_config = config.get_prepare_base_model_config()
        preprocess_data = DataPreprocessor(training_config)
        X, Y = preprocess_data.load_data()
        X_train, Y_train, X_val, Y_val,num_classes = preprocess_data.prepare_data(X, Y)    
        training = Training(training_config,prepare_base_model_config.params_learning_rate,X_train, Y_train, X_val, Y_val,)
        training.load_base_model()
        training.train()


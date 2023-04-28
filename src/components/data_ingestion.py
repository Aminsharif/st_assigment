import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    row_data_path:str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initial_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            current_folder = os.path.abspath('.')
            df=pd.read_csv(os.path.join(current_folder, 'notebooks','data','clean_data.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path, index=False)
            logging.info("Train test split")
            train_set, test_set= train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Excepton occured at data Ingestion stage')
            raise CustomException(e, sys)
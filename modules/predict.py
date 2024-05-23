import dill
import logging
import pandas as pd
import os
from datetime import datetime

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')


def predict() -> None:
    logging.info(f'Start Prediction')

    def sorted_directory_listing_by_creation_time(directory):
        def get_creation_time(item):
            item_path = os.path.join(directory, item)
            return os.path.getctime(item_path)

        items = os.listdir(directory)
        sorted_items = sorted(items, key=get_creation_time)
        return sorted_items[0]

    # Open .pkl file with saved pipeline with model
    file_model = f"{path}/data/models/" + sorted_directory_listing_by_creation_time(f"{path}/data/models/")
    logging.info(f'Start opening file: {file_model}')
    with open(file_model, 'rb') as pkl_pipe:
        model_pipe = dill.load(pkl_pipe)

    # Open all json files with testing data, save all testing data in new DataFrame
    path_jsons = f"{path}/data/test/"
    files_list = os.listdir(path_jsons)
    logging.info(f'Try Reading jsons files: {files_list}')
    testingdata = []
    for file in files_list:
        file_path = path_jsons + file
        file_series = pd.read_json(file_path, typ='series')
        testingdata.append(file_series)
    df_testingdata = pd.DataFrame(testingdata)  # Save all appended Series in one DataFrame

    # Get predictions by testing data and save to result DataFrame
    df_testingdata['prediction'] = model_pipe.predict(df_testingdata)

    # Save result DataFrame to csv file
    result_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_testingdata.to_csv(result_filename)

    print(df_testingdata[['id', 'prediction']])
    logging.info(f"Predictions: {df_testingdata[['id', 'prediction']]}")


if __name__ == '__main__':
    predict()

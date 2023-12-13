import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import pickle
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf
from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd


conf.set('core', 'enable_xcom_pickling', 'True')

LOCAL_PREPROCESS_FILE_PATH = '/tmp/preprocess.py'
GITHUB_PREPROCESS_RAW_URL = 'https://raw.githubusercontent.com/SowmyaC28/CreditCardFraudDetection_MLOps_Pipeline/main/src/data_preprocess.py'  # Adjust the path accordingly

LOCAL_TRAIN_FILE_PATH = '/tmp/train.py'
GITHUB_TRAIN_RAW_URL = 'https://raw.githubusercontent.com/SowmyaC28/CreditCardFraudDetection_MLOps_Pipeline/main/src/trainer/train.py'  # Adjust the path accordingly


def card_frequency(data):
    
    import pickle 
    import pandas as pd
    
    '''
        Calculates the  frequency of card usage
    '''
    df = pickle.loads(data)
    freq = df.groupby('cc_num').size()
    df['cc_freq'] = df['cc_num'].apply(lambda x : freq[x])
    
    pkl_df = pickle.dumps(df)
    return pkl_df

def encode_categorical_col(data):
    import pickle
    import pandas as pd
    from category_encoders import WOEEncoder
    ''' Encodes Categorical Col using WOE Encoder'''
    df = pickle.loads(data)
   
    for col in ['city','job','merchant', 'category']:
        df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud'])
        
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df 


def date_transformations(data):
    '''
        Converts date column to datetime format and 
        extracts 3 new columns 'day' ,'hour', 'month'
    '''
    import pickle
    import pandas as pd
    df = pickle.loads(data)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'],format='mixed')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.weekday
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df

def distance_calculation(data):
    '''
       Calculates distance between customer location
       and merchant location in km to a new column 'distance_km'.
       Drops 'lat','long','merch_lat','merch_long' columns
    '''
    import pickle
    import pandas as pd
    from geopy.distance import great_circle   
    df = pickle.loads(data)
    df['distance_km'] = df.apply(lambda col : round(great_circle((col['lat'],col['long']),
                                                (col['merch_lat'],col['merch_long'])).kilometers,2),axis=1)
    df.drop(columns=['lat','long','merch_lat','merch_long'],inplace=True)
    
    
   
    
    pkl_df = pickle.dumps(df)
    return pkl_df
    
def drop_col(data):
    ''' 
        Drops the unwanted columns and reorders the columns of DataFrame
    '''
    import pickle
    import pandas as pd
    df = pickle.loads(data)
    
    df.drop(columns=['cc_num','trans_date_trans_time','city_pop'],inplace=True)
    #Reorder columns
    df = df[['cc_freq','city','job','age','gender_M','merchant', 'category',
            'distance_km','month','day','hour','year','hours_diff_bet_trans','amt','is_fraud']]
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df

def extracting_age(data):
    '''
        Computes new column 'age' using the 'dob' column
        and drops the 'dob' column
    '''
    import pickle
    
    df = pickle.loads(data)

    df['dob'] = pd.to_datetime(df['dob'],format='mixed')
    df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year).astype(int)
    df.drop(columns='dob',inplace=True)
    
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df


def gender_ohe(data):
    '''
        One Hot Encodes the 'gender' colunm
    '''
    import pickle
    import pandas as pd
    df = pickle.loads(data)
    #Convert gender to binary classification
    df = pd.get_dummies(df,columns=['gender'],drop_first=True)
    
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df

def merchant_transformations(data):
    '''
       Cleans the 'merchant' column
    '''
    import pickle
    import pandas as pd
    df = pickle.loads(data)
    df['merchant'] = df['merchant'].apply(lambda x : x.replace('fraud_',''))
    
    
    pkl_df = pickle.dumps(df)
    return pkl_df

def transaction_gap(data):
    '''
        Calculates the time difference between each card usage in hours
        If the card is being used for the firt time, the difference is set to 0 for that particular instance
    '''
    import pickle 
    import pandas as pd
    import numpy as np
    df = pickle.loads(data)
    
    df.sort_values(['cc_num', 'trans_date_trans_time'],inplace=True)
    df['hours_diff_bet_trans']=((df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))
    df['hours_diff_bet_trans'].fillna(0,inplace=True)
    
   
    pkl_df = pickle.dumps(df)
    return pkl_df


def load_data():
    '''load data from gcs file'''
    import gcsfs
    import os
    from dotenv import load_dotenv
    

    # Load environment variables
    load_dotenv()

# Initialize variables
    fs = gcsfs.GCSFileSystem()
    storage_client = storage.Client()
    bucket_name = os.getenv("BUCKET_NAME")
    MODEL_DIR = os.environ['AIP_STORAGE_URI']
    gcs_train_data_path = "gs://mlops_pipeline/data/train/train_data.csv"
    with fs.open(gcs_train_data_path) as f:
        df = pd.read_csv(f)
    pkl_df = pickle.dumps(df)
    
    return pkl_df
    
def write_data(data):
    '''
        save mean into json file
        and save the cleaned data to cleaned_data.csv in the bucket
    '''
    
    import pandas as pd
    import pickle
    import gcsfs
    import json
    
    df = pickle.load(data)
    mean_train = df.drop('is_fraud',axis=1).mean()
    std_train = df.drop('is_fraud',axis=1).std()
    fs = gcsfs.GCSFileSystem()
    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }
    # Save the normalization statistics to a JSON file on GCS
    normalization_stats_gcs_path = "gs://mlops_pipeline/scaler/normalization_stats.json"
    with fs.open(normalization_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)
        
    file_path = "gs://mlops_pipeline/data/train/processed_data.csv"   
    with fs.open(file_path,'w') as file:
        df.to_csv(file,index=False) 
      
    pkl_df = pickle.dumps(df)
    return pkl_df
      
default_args = {
    'owner': 'Time_Series_IE7374',
    'start_date': dt.datetime(2023, 10, 24),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Model retraining at 9 PM everyday',
    schedule_interval='0 21 * * *',  # Every day at 9 pm
    catchup=False,
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)


date_column_task = PythonOperator(
    task_id = 'date_column_task',
    python_callable = date_transformations,
    op_args = [load_data_task.output],
    dag=dag,
)


merchant_column_task = PythonOperator(
    task_id = 'merchant_column_task',
    python_callable = date_transformations,
    op_args = [date_column_task.output],
    dag=dag,
)

dob_column_task = PythonOperator(
    task_id = 'dob_column_task',
    python_callable = extracting_age,
    op_args = [merchant_column_task.output],
    dag=dag,
)

distance_task = PythonOperator(
    task_id = 'distance_task',
    python_callable = distance_calculation,
    op_args = [dob_column_task.output],
    dag =dag,
)

ohe_task = PythonOperator(
    task_id = 'ohe_task',
    python_callable = gender_ohe,
    op_args = [distance_task.output],
    dag = dag,  
)

transaction_gap_task = PythonOperator(
    task_id = 'transaction_gap',
    python_callable = transaction_gap,
    op_args = [ohe_task.output],
    dag = dag, 
)

card_frequency_task = PythonOperator(
    task_id = 'card_frequency_task',
    python_callable = card_frequency,
    op_args = [transaction_gap_task.output],
    dag = dag, 
)

drop_task = PythonOperator(
    task_id = 'drop_task',
    python_callable = drop_col,
    op_args = [card_frequency_task.output],
    dag = dag,
)

categorical_columns_task = PythonOperator(
    task_id = 'categorical_columns_task',
    python_callable = encode_categorical_col,
    op_args = [drop_task.output],
    dag = dag, 
)

write_to_file = PythonOperator(
    task_id = 'write_to_file',
    python_callable = write_data,
    op_args=[categorical_columns_task.output],
    dag=dag,
)



# Tasks for pulling scripts from GitHub
pull_preprocess_script = BashOperator(
    task_id='pull_preprocess_script',
    bash_command=f'curl -o {LOCAL_PREPROCESS_FILE_PATH} {GITHUB_PREPROCESS_RAW_URL}',
    dag=dag,
)

pull_train_script = BashOperator(
    task_id='pull_train_script',
    bash_command=f'curl -o {LOCAL_TRAIN_FILE_PATH} {GITHUB_TRAIN_RAW_URL}',
    dag=dag,
)



env = {
    'AIP_STORAGE_URI': 'gs://mlops_pipeline/model'
}

# Tasks for running scripts
run_preprocess_script = BashOperator(
    task_id='run_preprocess_script',
    bash_command=f'python {LOCAL_PREPROCESS_FILE_PATH}',
    env=env,
    dag=dag,
)

run_train_script = BashOperator(
    task_id='run_train_script',
    bash_command=f'python {LOCAL_TRAIN_FILE_PATH}',
    env=env,
    dag=dag,
)

# Setting up dependencies
pull_preprocess_script >> pull_train_script >> run_preprocess_script >> load_data_task >> date_column_task >> merchant_column_task >> dob_column_task >> distance_task >> ohe_task >> transaction_gap_task >> card_frequency_task >> drop_task >> categorical_columns_task >> write_to_file>>run_train_script
# add preprocessing dags before 'run_train_script'
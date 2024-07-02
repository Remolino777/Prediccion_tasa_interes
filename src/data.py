from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import requests 

from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR, TRAIN_DATA_DIR



#url = 'https://www.banrep.gov.co/sites/default/files/Serie_historica_ipvnbr.xlsx'

def download_one_file_of_raw_data(url):      #_______________________  Load datasets
    response = requests.get(url)
    response.raise_for_status()    
    
    if response.status_code == 200:
        # Guardar el contenido del archivo .xlsx
        with open(RAW_DATA_DIR/'Serie_historica_ipvnbr.xlsx', 'wb') as f:
            f.write(response.content)
    
    else:
        print(f"No se pudo descargar el archivo. Código de estado: {response.status_code}")
    

def load_and_validate_data(ruta):  #_______LOAD AND VALIDATE DATA
    
    data = pd.read_excel(ruta,
                         sheet_name='Serie_IPVNBR',
                         header=2,
                         skiprows=2,
                         skipfooter=8)
    df = data[data['Unnamed: 0'] >= '2006-01-01']
    df1 = df[['Unnamed: 0','Bogotá.1','Alrededores de Bogotá4,5','Medellín.1','Cali.1']]
    df1 = df1.rename(columns={'Unnamed: 0':'Fecha',
                    'Bogotá.1':'Bogotá',
                    'Alrededores de Bogotá4,5':'Alrededores de Bogotá',
                    'Medellín.1':'Medellín',
                    'Cali.1':'Cali'
                    })
    #Exportar el archivo
    df1.to_parquet(TRANSFORMED_DATA_DIR/'validate_data.parquet')
 
    
def transform_raw_data(ruta): #_______TRANSFORM_RAW_DATA
    c_list = ['Bogotá', 'Alrededores de Bogotá', 'Medellín', 'Cali']
    df = pd.read_parquet(ruta)
    for i in c_list:
        df_place = df[['Fecha', i]]      
        df_place.to_parquet(f'{TRANSFORMED_DATA_DIR}/tes_{i}.parquet')
    

# Funcion para cortar indices y cerar los dataframes
def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = n_features
        subseq_last_idx = n_features + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices
 
ciudades = ['Bogotá', 'Alrededores de Bogotá', 'Medellín', 'Cali'] 
    
def transform_ts_data_to_features_and_target(ciudades):
        
    for i in ciudades:
        df_c = pd.read_parquet(f'{TRANSFORMED_DATA_DIR}/tes_{i}.parquet')
        df_c.reset_index(drop=True, inplace=True)  # Asegúrate de que los índices se restablecen

        n_features = 12
        step_size = 1

        indices = get_cutoff_indices(df_c, n_features, step_size) #Llama a la funcion externa
        
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, n_features), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        
        months = []

        for u, idx in enumerate(indices):
            x[u, :] = df_c.iloc[idx[0]:idx[1]][i].values
            y[u] = df_c.iloc[idx[1]:idx[2]][i].values  # Asegúrate de seleccionar la columna correcta
            months.append(df_c.iloc[idx[1]]['Fecha'])
            
        feature = pd.DataFrame(x, columns=[f'month_{j+1}' for j in range(n_features)])
        
        target = pd.DataFrame(y, columns=['Target'])
        
        # Crear el DataFrame con los datos concatenados.
        df_ready = pd.concat([feature, target], axis=1)
        
        # Exportar los datos al archivo train
        #ruta = r'D:\0_Respaldo\0_Proyectos_2024\ML_proyects\Prediccion tasa de interes\Prediccion_tasa_interes\data\Transform\train\train_'
        df_ready.to_parquet(f'{TRAIN_DATA_DIR}/train_{i}.parquet')
          
    
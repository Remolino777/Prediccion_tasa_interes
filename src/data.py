from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import requests 
from bs4 import BeautifulSoup

from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

#Variables

#url = 'https://www.datos.gov.co/Hacienda-y-Cr-dito-P-blico/Tasas-de-Inter-s-Activas-Informe-Semanal/yvb2-ppaa/about_data'

def download_one_file_of_raw_data(url):
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    csv_link_tag = soup.find('forge-dialog', href=True, string='Descargar')
    
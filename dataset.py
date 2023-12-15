import pandas as pd
import numpy as np

def load_data():
    data_path = 'data'
    ecog = f'{data_path}/ECoG.csv'
    motion = f'{data_path}/Motion.csv'

    ecog_df = pd.read_csv(ecog)
    motion_df = pd.read_csv(motion)

    return ecog_df, motion_df

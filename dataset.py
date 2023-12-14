#import torch
import pandas as pd
import numpy as np

'''
class K1Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.ecog, self.motion = load_data()

    def __len__(self):
        return len(self.motion)

    def __getitem__(self, idx):
        # sample - ECoG_Time
        sample = self.ecog.iloc[idx].values[1:]

        # Return Experimentor's hand position
        target = self.motion.iloc[idx].values[-3:]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def get_in_out_size(self):
        sample, target = self.__getitem__(0)

        return len(sample), len(target)
'''

def load_data():
    data_path = 'data'
    ecog = f'{data_path}/ECoG.csv'
    motion = f'{data_path}/Motion.csv'

    ecog_df = pd.read_csv(ecog)
    motion_df = pd.read_csv(motion)
    # ecog_df = ecog_df[ecog_df['ECoG_time'].isin(motion_df['MotionTime'])]

    return ecog_df, motion_df

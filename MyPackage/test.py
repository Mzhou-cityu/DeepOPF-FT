import numpy as np
import config
from test_model import test_model
import pandas as pd
from load_data import load_test_data

def test():
    system = 1
    data = np.zeros((10, config.Nsystem))
    data_PP = np.zeros((10, config.Nsystem))

    for i in range(Nsystem):
        load_data()

        data[:, i], data_PP[:, i] = test_model()

        system = system + 1

    data_df = pd.DataFrame(data)

    data_df.index = config.Evaluation_index
    data_df.columns = config.Test_case
    writer = pd.ExcelWriter('result' + config.mode + '.xlsx')
    data_df.to_excel(writer, float_format='%.2f')
    writer.save()

    data_df = pd.DataFrame(data_PP)
    data_df.index = config.Evaluation_index
    data_df.columns = config.Test_case
    writer = pd.ExcelWriter('resultPP_' + mode + '.xlsx')
    data_df.to_excel(writer, float_format='%.2f')
    writer.save()
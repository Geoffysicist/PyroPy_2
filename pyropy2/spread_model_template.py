import pandas as pd
import numpy as np
import helpers as h
from pathlib import Path
from fdrs_calcs import spread_models as sm

if __name__ == '__main__':

    # read weather to df and rename to fdrs_calcs compatible names
    test_weather_data_pth = Path('pyropy2/.data/point_forecast.csv')
    df = pd.read_csv(test_weather_data_pth, 
                     header=h.find_header_row(test_weather_data_pth, 'date')
                     )

    new_column_names = {
        'Temp (C)': 'air_temperature',
        'RH (%)': 'relative_humidity',
        'Wind Dir': 'wind_direction',
        'Wind Speed (km/h)': 'wind_speed',
        'Wind max in hr km/h': 'wind_speed_max',
        'Drought Factor': 'DF',
        'FBI': 'ADFD_FBI',
    }

    df = df.rename(columns=new_column_names)

    #  create a time column
    df['time'] = pd.to_datetime(df['Local Date']+ ' ' + df['Local Time'])

    #  model specific stuff

    print(df.head())
import pandas as pd
import numpy as np
from pathlib import Path
from fdrs_calcs import spread_models as sm

def calc_spread_probability(
    wind_speed, 
    wind_adjustment_factor,
    fuel_moisture, 
    fuel_load_near_surface,
):
    '''Calculate the spread probability based on Cruz using 2m windspeed and 
    near surface fuel load
    '''
    spread_probability = 1 / (
        1
        + np.exp(
            -(
                9.0787
                + 0.7150 * wind_speed * wind_adjustment_factor
                - 2.2325 * fuel_moisture
                + 14.8674 * fuel_load_near_surface
            )
        )
    )
    return spread_probability

if __name__ == '__main__':
    import helpers as h
else:
    from pyropy2 import helpers as h

if __name__ == '__main__':

    cover = 20
    overstorey_fuel_height = 2
    wind_adjustment_factor = 0.5
    precipitation = 0
    time_since_rain = 48
    fuel_load_near_surface = 2

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
    df['fuel_moisture'] = sm.mallee_heath.calc_fuel_moisture(
        df.relative_humidity,
        df.air_temperature,
        (df.time.dt.month,df.time.dt.hour),
        precipitation,
        time_since_rain,       
    )

    df['spread_probability'] = calc_spread_probability(
        df.wind_speed,
        wind_adjustment_factor,
        df.fuel_moisture,
        fuel_load_near_surface,
    )

    # calculate the probability of crown fire
    df['crown_probability'] = sm.mallee_heath.calc_crown_probability(df.wind_speed, df.fuel_moisture)

    df['rate_of_spread_surface'] = (
        60 * #convert m/min to m/h
        3.337 * df.wind_speed
        * np.exp(-0.1284*df.fuel_moisture)
        * np.power(overstorey_fuel_height, -0.7073)
    )

    df['rate_of_spread_crown'] = (
        60 * #convert m/min to m/h
        9.5751 * df.wind_speed
        * np.exp(-0.1795*df.fuel_moisture)
        * np.power(0.01*cover,  0.3589)
    )

    df['rate_of_spread'] = np.where(
        df.spread_probability > 0.5,
        (1.0-df['crown_probability'])*df['rate_of_spread_surface']+df['crown_probability']*df['rate_of_spread_crown'],
        0
    )

    print(df.head())
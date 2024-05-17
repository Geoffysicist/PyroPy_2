import pandas as pd
import numpy as np

from pathlib import Path
from fdrs_calcs import spread_models as sm

if __name__ == '__main__':
    import helpers as h
else:
    from pyropy2 import helpers as h

def calc_moisture_function(fuel_moisture):
    return np.where(
        fuel_moisture <= 4.1,
        1,
        np.where(
            fuel_moisture > 24,
            0,
            (
                0.9082 + 
                0.1206 * fuel_moisture - 
                0.03106 * np.power(fuel_moisture,2) + 
                0.001853 * np.power(fuel_moisture,3) - 
                0.00003467 * np.power(fuel_moisture,4)
            )
        )
    )

def calc_fuel_availability(DF, DI = 100, wrf = 3, wet_forest=False):
    ''' returns the fuel availability, that is the proporton of fuel available
    tp be burn in the fire front as per Cruz et al. 2022
    
    args
      DF: Drought factor
      DI: drought index - KBDI except SDI in Tas
      waf: wind adjustment factor between 3 and 5
      wet_forest: optional, if `True`, calculate the fuel availability
        for wet forests instead of dry forests. Default is `False`.
    '''
    if wet_forest:
        C1 = (
            (0.0046 * np.power(wrf, 2) - 0.0079 * wrf - 0.0175) * DI + 
            (-0.9167 * np.power(wrf, 2) + 1.5833 * wrf + 13.5)
            )
        C2 = 0 # TODO: implement slope/aspect effect
        drought_factor = DF * max(C1 + C2, 0) / 10
        drought_factor = np.clip(drought_factor, 0, 10)
    else:
        drought_factor = DF

    return 1.008 / (1 + 104.9 * np.exp(-0.9306 * drought_factor))

def calc_slope_effect(slope):
    ''' returns the slope steepness effect as per Cruz 2021 eq 13
    
    args:
      slope: slope gradient in degrees
    '''
    return np.where(
        slope == 0,
        1,
        np.where(
            slope > 0,
            np.power(2, slope/10),
            np.power(2,-slope/10)/(2 * np.power(2, -slope/10) - 1)
        )
    )

def calc_height_understorey(FHS_elevated, height_elevated):
    '''returns the average understorey height mas per Cruz 2021 eq 1
    
    args:
      FHS_elevated - elevated fuel hazard score
      elevatedfuel height (m)
    '''

    return -0.1 + 0.06 * FHS_elevated + 0.48 * height_elevated

def calc_ros_phase1(wind_speed, fuel_moisture, fuel_availability, fuel_load_surface, wrf, slope = 0):
    ''' returns the phase 1 forward rate of spread (km/h) from Cruz 2021 eqn 14a and b
    
    args:
      wind_speed (array_like) - 10m open wind speed (km/h)
      fuel_moisture - fine fuel moisture content %
      fuel_availability (array_like) - output of calc_fuel_availability
      fuel_load_surface - surface + near-surface fuel load t/ha
      wrf - wind reduction factor
      slope (optional): slope gradient in degrees, default is 0 
    '''
    fuel_moisture_effect = calc_moisture_function(fuel_moisture) * fuel_availability
    slope_effect = calc_slope_effect(slope)
    u = wind_speed / wrf

    ros = np.where(
        u > 2,
        0.03 + 0.05024 * np.power(u - 1, 0.92628) * np.power(fuel_load_surface / 10, 0.79928),
        0.03 
    )

    return ros * fuel_moisture_effect * slope_effect * 1000

def calculate_ros_phase2(wind_speed, fuel_moisture, fuel_availability, fuel_load_surface, wrf, height_understorey, slope = 0):
    ''' returns the phase 2 forward rate of spread (km/h) from Cruz 2021 eqn 15
    
    args:
      wind_speed (array_like) - 10m open wind speed (km/h)
      fuel_moisture - fine fuel moisture content %
      fuel_availability (array_like) - output of calc_fuel_availability
      fuel_load_surface - surface + near-surface fuel load t/ha
      wrf - wind reduction factor
      height_understorey - weighted average fo near-surface and elevated fuel heights (m)
      slope (optional): slope gradient in degrees, default is 0 
    '''
    fuel_moisture_effect = calc_moisture_function(fuel_moisture) * fuel_availability
    slope_effect = calc_slope_effect(slope)
    u = wind_speed / wrf

    ros = 0.19591 * np.power(u, 0.8257) * np.power(fuel_load_surface / 10, 0.4672) * np.power(height_understorey, 0.495)
    return ros * fuel_moisture_effect * slope_effect * 1000

def calc_ros_phase3(wind_speed, fuel_moisture, fuel_availability):
    '''retruns the phase 3 forward rate of spread (km/h) as per Cruz 2021 eq 16.
    Note decision to ignore slope factor for phase 3 fires
    
    args:
      wind_speed (array_like) - 10m open wind speed (km/h)
      fuel_moisture - fine fuel moisture content %
      fuel_availability (array_like) - output of calc_fuel_availability
    '''

    fuel_moisture_effect = calc_moisture_function(fuel_moisture) * fuel_availability

    ros = 0.05235 * np.power(wind_speed, 1.19128)
    return ros * fuel_moisture_effect * 1000

def calc_probability_phase2(wind_speed, fuel_moisture, fuel_availability, fuel_load_surface, wrf):
    ''' returns the probability of transition to phase 2 as per Cruz et al 2021 Eqn 9 and 10

    args:
      wind_speed (array_like) - 10m open wind speed (km/h)
      fuel_moisture - fine fuel moisture content %
      fuel_availability (array_like) - output of calc_fuel_availability
      fuel_load_surface - surface + near-surface fuel load t/ha
      wrf - wind reduction factor
    '''
    fuel_moisture_effect = calc_moisture_function(fuel_moisture) * fuel_availability

    g_x = -23.9315 + 1.7033 * (wind_speed / wrf) + 12.0822 * fuel_moisture_effect + 0.95236 * fuel_load_surface

    return np.where(
        fuel_load_surface < 1,
        0,
        1 / (1 + np.exp(-g_x))
    )

def calc_probability_phase3(wind_speed, fuel_moisture, fuel_availability, ros_phase2):
    '''returns the probability of transition to phase 3 as per Cruz 2021 eq 11 & 12
    
    args:
      wind_speed (array_like) - 10m open wind speed (km/h)
      fuel_moisture - fine fuel moisture content %
      fuel_availability (array_like) - output of calc_fuel_availability
      ros_phase2: predicted phase 2 forward rate of spread (m/h)     
    '''
    fuel_moisture_effect = calc_moisture_function(fuel_moisture) * fuel_availability

    g_x = -32.3074 + 0.2951 * wind_speed + 26.8734 * fuel_moisture_effect
    return np.where(
        ros_phase2 < 300, #note m/h
        0,
        1 / (1 + np.exp(-g_x))
    )

def calc_rate_of_spread(ros_phase1, ros_phase2, ros_phase3, probability_phase2, probability_phase3):
    '''returnsthe overall rate of spread as per Cruz 2021 eq 17'''

    return np.where(
        probability_phase2 < 0.5,
        ros_phase1 * (1 - probability_phase2) + ros_phase2 * probability_phase2,
        ros_phase1 * (1 - probability_phase2) + ros_phase2 * probability_phase2 * (1 - probability_phase3) + ros_phase3 * probability_phase3
    )

if __name__ == '__main__':

    wet_forest = False
    wrf = 3
    fuel_load_surface = 15
    slope = 0
    FHS_elevated = 3
    height_elevated = 2
    height_understorey = calc_height_understorey(FHS_elevated,height_elevated)

    print(f'height understorey: {height_understorey}')

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

    #  add fuel moisture
    df['fuel_moisture'] = sm.dry_forest.fuel_moisture_model(
        df.air_temperature,
        df.relative_humidity,
        (df.time.dt.month,df.time.dt.hour),
        wet_forest=wet_forest
    )

    # df['fuel_moisture_function'] = calc_moisture_function(df.fuel_moisture)

    df['fuel_availability'] = calc_fuel_availability(df.DF, wet_forest=wet_forest)

    df['ros_phase1'] = calc_ros_phase1(
        df.wind_speed, df.fuel_moisture, df.fuel_availability, fuel_load_surface, wrf, slope = slope
        )
    
    df['ros_phase2'] = calculate_ros_phase2(
        df.wind_speed, df.fuel_moisture, df.fuel_availability, fuel_load_surface, wrf, height_understorey, slope = slope
        )
    
    df['ros_phase3'] = calc_ros_phase3(df.wind_speed, df.fuel_moisture, df.fuel_availability)
    
    df['probability_phase2'] = calc_probability_phase2(
        df.wind_speed, df.fuel_moisture, df.fuel_availability, fuel_load_surface, wrf
        )
    
    df['probability_phase3'] = calc_probability_phase3(df.wind_speed, df.fuel_moisture, df.fuel_availability, df.ros_phase2)

    df['ROS'] = calc_rate_of_spread(
        df.ros_phase1, df.ros_phase2, df.ros_phase3, df.probability_phase2, df.probability_phase3
        )
    
    print(df.head())

    df.to_csv('test_output.csv')

    
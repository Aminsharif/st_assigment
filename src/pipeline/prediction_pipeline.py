import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import datetime


def get_weekday_from_date(date_str):

    date_str = '23-04-2023' # date in yyyy-mm-dd format
    date = datetime.datetime.strptime(date_str, '%d-%m-%Y')
    week_day = date.strftime('%A') # %A returns the full weekday name

    return week_day

def get_hour(time_str):
    time_obj = datetime.datetime.strptime(time_str, '%H:%M') # convert to datetime object
    hours = time_obj.hour
    return hours

def get_minute(time_str):
    time_obj = datetime.datetime.strptime(time_str, '%H:%M') # convert to datetime object
    minutes = time_obj.minute
    return minutes

import math

def distance(lat1, lon1, lat2, lon2):
    if isinstance(lat1, tuple):
        lat1 = lat1[0]
        lat1 = float(lat1)

    if isinstance(lon1, tuple):
        lon1 = lon1[0]
        lon1 = float(lon1)

    if isinstance(lat2, tuple):
        lat2 = lat2[0]
        lat2 = float(lat2)

    if isinstance(lon2, tuple):
        lon2 = lon2[0]
        lon2 = float(lon2)
    # Radius of the Earth in kilometers
    R = 6371
    print(lat1,'--', lon1, '--', lat2, '--', lon2)
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate the differences between the latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the distance using the Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 age:int,
                 ratting:float,
                 restaurant_latitude:float,
                 restaurant_longitude:float,
                 delivery_location_latitude:float,
                 delivery_location_longitude:float,
                 weather:str,
                 traffic:str,
                 vehicle_condition:int,
                 order_type:str,
                 type_of_vehicle:str,
                 multiple_deliver:int,
                 festival:str,
                 city:str,
                 ordered_date:str,
                 time_orderd:str,
                 time_order_picked:str):
        
        self.age=age
        self.ratting=ratting
        self.restaurant_latitude=restaurant_latitude,
        self.restaurant_longitude=restaurant_longitude,
        self.delivery_location_latitude=delivery_location_latitude,
        self.delivery_location_longitude=delivery_location_longitude,
        self.weather=weather
        self.traffic=traffic
        self.vehicle=vehicle_condition
        self.order_type=order_type
        self.type_of_vehicle = type_of_vehicle
        self.multiple_deliver = multiple_deliver
        self.festival = festival
        self.ordered_date = ordered_date,
        self.city=city
        self.time_orderd = time_orderd
        self.time_order_picked = time_order_picked

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.age],
                'Delivery_person_Ratings':[self.ratting],
                'Weather_conditions':[self.weather],
                'Road_traffic_density':[self.traffic],
                'Vehicle_condition':[self.vehicle],
                'Type_of_order':[self.order_type],
                'Type_of_vehicle':[self.type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliver],
                'Festival':[self.festival],
                'City':[self.city],
                'Week_days':[get_weekday_from_date(self.ordered_date)],
                'Time_Orderd_Hours':[get_hour(self.time_orderd)],
                'Time_Orderd_Minutes':[get_minute(self.time_orderd)],
                'Time_Order_picked_Hours':[get_hour(self.time_order_picked)],
                'Time_Order_picked_Minutes':[get_minute(self.time_order_picked)],
                'Distance_Resturant_to_Location':[distance(self.restaurant_latitude, self.restaurant_longitude, self.delivery_location_latitude, self.delivery_location_longitude)],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

from datetime import datetime

import joblib
import numpy as np
from featuretools import variable_types as vtypes
from featuretools.primitives import make_trans_primitive, TransformPrimitive
from featuretools.variable_types import LatLong, Boolean
import featuretools as ft
import pandas as pd

import taxi_utils
from ft_process import FeatureToolProcess


def haversine(latlong1, latlong2):
    lat_1s = np.array([x[0] for x in latlong1])
    lon_1s = np.array([x[1] for x in latlong1])
    lat_2s = np.array([x[0] for x in latlong2])
    lon_2s = np.array([x[1] for x in latlong2])
    lon1, lat1, lon2, lat2 = map(np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6367 * 2 * np.arcsin(np.sqrt(a))
    return km


def cityblock(latlong1, latlong2):
    lon_dis = haversine(latlong1, latlong2)
    lat_dist = haversine(latlong1, latlong2)
    return lon_dis + lat_dist


def bearing(latlong1, latlong2):
    lat1 = np.array([x[0] for x in latlong1])
    lon1 = np.array([x[1] for x in latlong1])
    lat2 = np.array([x[0] for x in latlong2])
    lon2 = np.array([x[1] for x in latlong2])
    delta_lon = np.radians(lon2 - lon1)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    x = np.cos(lat2) * np.sin(delta_lon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return np.degrees(np.arctan2(x, y))


Bearing = make_trans_primitive(function=bearing,
                               input_types=[LatLong, LatLong],
                               commutative=True,
                               return_type=vtypes.Numeric)

Cityblock = make_trans_primitive(function=cityblock,
                                 input_types=[LatLong, LatLong],
                                 commutative=True,
                                 return_type=vtypes.Numeric)


def is_rush_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 7) & (hour <= 11)


def is_noon_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 11) & (hour <= 13)


def is_night_hour(datetime):
    hour = pd.DatetimeIndex(datetime).hour
    return (hour >= 18) & (hour <= 23)


RushHour = make_trans_primitive(function=is_rush_hour,
                                input_types=[vtypes.Datetime],
                                return_type=vtypes.Boolean)

NoonHour = make_trans_primitive(function=is_noon_hour,
                                input_types=[vtypes.Datetime],
                                return_type=vtypes.Boolean)

NightHour = make_trans_primitive(function=is_night_hour,
                                 input_types=[vtypes.Datetime],
                                 return_type=vtypes.Boolean)


class GeoBox(TransformPrimitive):
    name = "GeoBox"
    input_types = [LatLong]
    return_type = Boolean

    def __init__(self, bottomleft, topright):
        super().__init__()
        self.bottomleft = bottomleft
        self.topright = topright

    def get_function(self):
        def geobox(latlong, bottomleft=self.bottomleft, topright=self.topright):
            lat = np.array([x[0] for x in latlong])
            lon = np.array([x[1] for x in latlong])
            boxlats = [bottomleft[0], topright[0]]
            boxlongs = [bottomleft[1], topright[1]]
            output = []
            for i, name in enumerate(lat):
                if (min(boxlats) <= lat[i] <= max(boxlats) and
                        min(boxlongs) <= lon[i] <= max(boxlongs)):
                    output.append(True)
                else:
                    output.append(False)
            return output

        return geobox

    def generate_name(self, base_feature_names):
        return u"GEOBOX({}, {}, {})".format(base_feature_names[0],
                                            str(self.bottomleft),
                                            str(self.topright))

matrix = FeatureToolProcess(1, 2016, 4, 11, 9, 31, 9, False, 1,
                            '(-73.88377, 40.83723)',
                            '(-73.88377, 40.83723)').feature_process()

model = joblib.load('model_proved.model')
print(taxi_utils.predict_xgb(model, matrix))
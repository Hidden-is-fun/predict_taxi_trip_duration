import sys
import time

import joblib
import numpy as np
import pandas as pd
from featuretools import variable_types as vtypes
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QLineEdit, QPushButton, QLabel, QButtonGroup, QRadioButton
from featuretools.primitives import make_trans_primitive
from featuretools.variable_types import LatLong, Boolean
import ft_process
from featuretools.primitives import make_trans_primitive, TransformPrimitive

import taxi_utils


class Test(QWidget):
    def __init__(self):
        super(Test, self).__init__()

        self.MainWidget = QWidget(self)
        self.setFixedSize(450, 450)

        _ = QLabel('起始坐标', self.MainWidget)
        _.setGeometry(30, 25, 100, 50)

        __ = QLabel('终点坐标', self.MainWidget)
        __.setGeometry(30, 90, 100, 50)

        ___ = QLabel('日期时间           -                        :        :', self.MainWidget)
        ___.setGeometry(30, 155, 500, 50)

        ____ = QLabel("车辆类型", self.MainWidget)
        ____.setGeometry(30, 203, 100, 50)

        _____ = QLabel("乘客数量", self.MainWidget)
        _____.setGeometry(30, 250, 100, 50)

        self.startPos = QLineEdit(self.MainWidget)
        self.startPos.setStyleSheet("font-size:20px;"
                                    "font-family:JetBrains Mono;")
        self.startPos.setGeometry(100, 25, 300, 50)

        self.endPos = QLineEdit(self.MainWidget)
        self.endPos.setStyleSheet("font-size:20px;"
                                  "font-family:JetBrains Mono;")
        self.endPos.setGeometry(100, 90, 300, 50)

        self.btn = QPushButton("预测", self.MainWidget)
        self.btn.setStyleSheet("font-size:15px;"
                               "font-family:Microsoft Yahei;")
        self.btn.setGeometry(100, 315, 140, 50)
        self.btn.clicked.connect(self.predict)

        self.btn1 = QPushButton("获取当前时间", self.MainWidget)
        self.btn1.setStyleSheet("font-size:15px;"
                                "font-family:Microsoft Yahei;")
        self.btn1.setGeometry(260, 315, 140, 50)
        self.btn1.clicked.connect(self.get_time)

        spacer = [0, -10, 20, 10, 0]
        self.time = []
        for _i in range(5):
            time = QLineEdit(self.MainWidget)
            time.setGeometry(100 + _i * 65 + spacer[_i], 155, 40, 50)
            time.setStyleSheet("font-size:20px;"
                               "font-family:JetBrains Mono;")
            time.setAlignment(Qt.AlignCenter)
            self.time.append(time)

        self.selector1 = QRadioButton('Normal', self.MainWidget)
        self.selector2 = QRadioButton('Deluxe', self.MainWidget)
        self.selector1.move(100, 220)
        self.selector2.move(200, 220)
        self.selector1.setStyleSheet("font-size:15px;")
        self.selector2.setStyleSheet("font-size:15px;")
        self.result = QLabel(self.MainWidget)
        self.result.setGeometry(100, 370, 200, 50)
        self.result.setStyleSheet("font-size:20px;"
                                  "font-family:JetBrains Mono;"
                                  "color:red;")
        self.result.setWordWrap(True)

        self.num = QLineEdit(self.MainWidget)
        self.num.setGeometry(100, 250, 300, 50)
        self.num.setStyleSheet("font-size:20px;"
                               "font-family:JetBrains Mono;")

    def get_time(self):
        localtime = time.localtime(time.time())
        print(f'{localtime[3]}:{localtime[4]}:{localtime[5]}')
        for _i in range(5):
            self.time[_i].setText(str(localtime[_i + 1]))

    def predict(self):
        if self.startPos.text() and self.endPos.text():
            '''
            sx = self.startPos.text()[1:9]
            sy = self.startPos.text()[11:19]
            ex = self.endPos.text()[1:9]
            ey = self.endPos.text()[11:19]
            res = predict([[1 if self.selector1.isChecked() else 2,
                            int(self.num.text()),
                            -float(sy),
                            float(sx),
                            -float(ey),
                            float(ex),
                            False,
                            0,
                            int(self.time[0].text()),
                            int(self.time[1].text()),
                            int(self.time[2].text()),
                            int(self.time[3].text()),
                            int(self.time[4].text())]])
            self.result.setText(str(res))
            '''
            _matrix = ft_process.FeatureToolProcess(
                int(self.num.text()),
                2016,
                int(self.time[0].text()),
                int(self.time[1].text()),
                int(self.time[2].text()),
                int(self.time[3].text()),
                int(self.time[4].text()),
                False,
                1 if self.selector1.isChecked() else 2,
                self.startPos.text(),
                self.endPos.text(),
            ).feature_process()
            _ = joblib.load('model_proved.model')
            res = taxi_utils.predict_xgb(_, _matrix)["trip_duration"][0]
            self.result.setText(str(res))
        else:
            self.result.setText("请输入坐标")


if __name__ == '__main__':
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


    matrix = ft_process.FeatureToolProcess(1, 2016, 4, 11, 9, 31, 9, False, 1,
                                           '(-73.88377, 40.83723)',
                                           '(-73.88377, 40.83723)').feature_process()

    model = joblib.load('model_proved.model')
    print(taxi_utils.predict_xgb(model, matrix))
    app = QApplication(sys.argv)
    demo = Test()
    demo.show()
    sys.exit(app.exec_())

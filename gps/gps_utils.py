##########################################################################################################
#                                                                                                        #
#   GPS utilities (gps3 functionality)                                                                   #
#   file: gps_utils.py                                                                                   #
#                                                                                                        #
#   Author: Javier Goya Pérez                                                                            #
#   Date: January 2021                                                                                   #
#                                                                                                        #
##########################################################################################################
from gps3 import gps3
import sys

def init_gps():
    gps_socket = gps3.GPSDSocket()
    data_stream = gps3.DataStream()
    gps_socket.connect()
    gps_socket.watch()
    return gps_socket, data_stream

def get_position(gps_socket, data_stream):
    lat = lon = 0
    for new_data in gps_socket:
        if new_data:
            data_stream.unpack(new_data)
            lat = data_stream.TPV["lat"]
            lon = data_stream.TPV["lon"]
    return lat, lon
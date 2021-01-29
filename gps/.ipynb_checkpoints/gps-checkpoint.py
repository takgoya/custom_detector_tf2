from gps3 import gps3
import sys

gps_socket = gps3.GPSDSocket()
data_stream = gps3.DataStream()
gps_socket.connect()
gps_socket.watch()

try:
    for new_data in gps_socket:
        if new_data:
            data_stream.unpack(new_data)
            print("[INFO] GPS Time = ", data_stream.TPV["time"])
            print("[INFO] GPS Latitud = ", data_stream.TPV["lat"])
            print("[INFO] GPS Longitud = ", data_stream.TPV["lon"])
            print("[INFO] GPS Altitud = ", data_stream.TPV["alt"])
        #else:
            #print("[INFO] GPS not receiving data")

except KeyboardInterrupt:
    sys.exit(0)
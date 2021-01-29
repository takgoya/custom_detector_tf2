##########################################################################################################
#                                                                                                        #
#   DB utilities (sqlite3 functionality)                                                                  #
#   file: db_utils.py                                                                                    #
#                                                                                                        #
#   Author: Javier Goya Pérez                                                                            #
#   Date: January 2021                                                                                   #
#                                                                                                        #
##########################################################################################################

# This code is based on the SQLite tutorial
# https://www.sqlitetutorial.net/sqlite-python/

import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """
    create a database connection to a SQLite database
    """
    
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print("[INFO] DB::create_connection error = {}".format(e))
    
    return conn

def create_recordings_table(conn):
    """
    create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    
    sql_create_recordings_table = """CREATE TABLE IF NOT EXISTS recordings (
                                    id INTEGER PRIMARY KEY,
                                    name TEXT NOT NULL);"""
    
    try:
        c = conn.cursor()
        c.execute(sql_create_recordings_table)
    except Error as e:
        print("[INFO] DB::create_recordings_table error = {}".format(e))

def create_detections_table(conn):
    """
    create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    
    sql_create_detections_table = """CREATE TABLE IF NOT EXISTS detections (
                                id INTEGER PRIMARY KEY,
                                recording_id INTEGER NOT NULL,
                                object_name TEXT NOT NULL,
                                licence TEXT NOT NULL,
                                gps_lat REAL,
                                gps_lon REAL,
                                elapsed_time REAL NOT NULL,
                                frame INTEGER NOT NULL,
                                datetime text NOT NULL,
                                FOREIGN KEY (recording_id) REFERENCES recordings (id),
                                unique(object_name, licence, datetime));"""
    
    try:
        c = conn.cursor()
        c.execute(sql_create_detections_table)
    except Error as e:
        print("[INFO] DB::create_detections_table error = {}".format(e))
        
def insert_recording(conn, recording):
    """
    Create a new recording into the recordings table
    :param conn:
    :param recording:
    :return: recording id
    """
    
    sql = """INSERT INTO recordings(name) VALUES(?);"""
     
    cursor = conn.cursor()
    cursor.execute(sql, (recording,))
    conn.commit()
    return cursor.lastrowid


def insert_detection(conn, detection):
    """
    Create a new detection
    :param conn:
    :param detection:
    :return:
    """
    
    sql = """INSERT OR IGNORE INTO detections(recording_id, object_name, licence, gps_lat, gps_lon, elapsed_time, frame, datetime) VALUES(?,?,?,?,?,?,?,?);"""
    cursor = conn.cursor()
    cursor.execute(sql, detection)
    conn.commit()
    return cursor.lastrowid
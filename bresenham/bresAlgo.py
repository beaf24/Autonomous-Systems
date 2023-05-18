import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def determine_coords(x0, y0, z_angle, laser_angle, laser_measure, meters_pixel_ratio):
    line_angle = z_angle + laser_angle

    if( abs(line_angle) > 2*math.pi ):
        if( line_angle > 0):
            line_angle = line_angle - 2*math.pi
        else:
            line_angle = line_angle + 2*math.pi
    print(line_angle*180/math.pi)
    
    x1 =  x0 + (laser_measure * math.cos(line_angle) ) / meters_pixel_ratio
    y1 =  y0 + (laser_measure * math.sin(line_angle) ) / meters_pixel_ratio
    print(y1)

    return [ int(x1), int(y1) ]


def bres_algo(x0, y0, x1, y1):
    pointX = x0
    pointY = y0
    bresenham_points = []

    dX = abs( x1 - x0 )
    dY = -abs( y1 - y0 )
    increment_X = None
    increment_Y = None
    error =  dX + dY

    
    if( x0 < x1 ):
        increment_X = 1
    else:
        increment_X = -1

    if( y0 < y1 ):
        increment_Y = 1
    else:
        increment_Y = -1

    while(True):
        bresenham_points.append([pointX, pointY])

        if ( pointX == x1) and ( pointY == y1):
            break

        e2 = error*2

        if ( e2 >= dY ):
            if ( pointX == x1 ):
                break

            error = error + dY
            pointX = pointX + increment_X

        if ( e2 <= dX ):
            if ( pointY == y1):
                break
            
            error = error + dX
            pointY = pointY + increment_Y

    return bresenham_points

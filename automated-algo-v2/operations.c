//
// Created by herri on 22/05/2023.
//

#include "operations.h"
#include <math.h>
#include <stdio.h>

double z_quat_to_euler( float w, float z){
    double t3 = 2 * (w * z);
    double t4 = 1.0f - 2.0 * (z * z);

    return  atan2(t3, t4);
}

int determine_x_coord( int x0, double robot_orient, double laser_angle, double laser_measure, double ratio ){
    double line_angle = robot_orient + laser_angle;
    return (int) (x0 + (laser_measure * cos(line_angle) ) / ratio);

}

int determine_y_coord( int y0, double robot_orient, double laser_angle, double laser_measure, double ratio ){
    double line_angle = robot_orient + laser_angle;

    return (int) (y0 + (laser_measure * sin(line_angle) ) / ratio);

}

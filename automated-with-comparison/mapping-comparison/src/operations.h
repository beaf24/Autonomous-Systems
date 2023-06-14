//
// Created by herri on 22/05/2023.
//

#ifndef MAPPING_OPERATIONS_H
#define MAPPING_OPERATIONS_H

double z_quat_to_euler( float w, float z);
int determine_x_coord( int x0, double robot_orient, double laser_angle, double laser_measure, double ratio );
int determine_y_coord( int y0, double robot_orient, double laser_angle, double laser_measure, double ratio );

#endif //MAPPING_OPERATIONS_H

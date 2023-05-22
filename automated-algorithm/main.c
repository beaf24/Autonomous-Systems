#include <stdio.h>
#include <stdlib.h>
#include "operations.h"
#include <math.h>


int main(int argc, char **argv) {

    if( argc != 2){
        printf("ERROR: Missing meters/pixel ratio argument\n");
        exit(1);
    }
    FILE *file = fopen("bag_data.txt", "r");

    FILE *export = fopen("export.txt", "a");

    

    
    double meter_per_pixel;
    sscanf(argv[1], "%lf", &meter_per_pixel);

    double l_occ = log(0.7/0.3);
    double l_free = log(0.3/0.7);

    int x_max = 0, x_min = 0;
    int y_max = 0, y_min = 0;

    if(!file) return 1;
    if(!export) return 2;

    float x, y, quat_w, quat_z, angle, measure;
    double robot_orientation = 0;
    int robot_x = 0, robot_y = 0;


    // Read data
    while(fscanf(file, "%f %f %f %f %f %f ", &x, &y, &quat_w, &quat_z, &angle, &measure) == 6){

        if( (int) (measure/meter_per_pixel) == 0) continue;
        int obs_x = 0, obs_y = 0;
        // Determine robot's orientation and coordinates
        robot_orientation = z_quat_to_euler( quat_w, quat_z);
        robot_x = (int) (x/meter_per_pixel);
        robot_y = (int) (y/meter_per_pixel);


        // Set new x_max or x_min
        if(robot_x > x_max) x_max = robot_x;
        else if( robot_x < x_min) x_min = robot_y;

        // Set new y_max or y_min
        if(robot_y > y_max) y_max = robot_y;
        else if( robot_y < y_min) y_min = robot_y;

        // Determine detected obstacle coordinates
        obs_x = determine_x_coord(robot_x, robot_orientation, angle, measure, meter_per_pixel);
        obs_y = determine_y_coord(robot_y, robot_orientation, angle, measure, meter_per_pixel);

        // Set new x_max or x_min
        if(obs_x > x_max) x_max = obs_x;
        else if( obs_x < x_min) x_min = obs_x;

        // Set new y_max or y_min
        if(obs_y > y_max) y_max = obs_y;
        else if( obs_y < y_min) y_min = obs_y;

        // Bresenham
        int point_x = robot_x;
        int point_y = robot_y;

        int dx = abs(obs_x - robot_x);
        int dy = -abs(obs_y - robot_y);
        int inc_x = 0, inc_y = 0;
        int error = dx + dy;

        if (robot_x < obs_x) inc_x = 1;
        else inc_x = -1;

        if (robot_y < obs_y) inc_y = 1;
        else inc_y = -1;

        while(1){
            if (point_x == obs_x && point_y == obs_y){
                fprintf(export, "%d %d %f\n", point_x, point_y, l_occ);
                break;
            }

            fprintf(export, "%d %d %f\n", point_x, point_y, l_free);

            int e2 = error * 2;

            if (e2 >= dy){
                if(point_x == obs_x) break;
                error = error + dy;
                point_x = point_x + inc_x;
            }

            if(e2 <= dx){
                if(point_y == obs_y) break;
                error = error + dx;
                point_y = point_y + inc_y;
            }
        }

        // END of Bresenham //

    }
    // END of Read Data //


    fclose(export);
    fclose(file);

    FILE *size = fopen("size.txt", "w");

//    int max_size_x = x_max + abs(x_min);
//    int max_size_y = y_max + abs(y_min);

    fprintf(size ,"%d %d %d %d", x_max, x_min, y_max, y_min);
    fclose(size);

    return 0;
}

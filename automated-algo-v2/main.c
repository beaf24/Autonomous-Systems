#include <stdio.h>
#include <stdlib.h>
#include "operations.h"
#include <math.h>

typedef struct Node
{
    int x;
    int y;
    double l_odd;
    struct Node* next;
    
}node_t;

// double z_quat_to_euler( float w, float z){
//     double t3 = 2 * (w * z);
//     double t4 = 1.0f - 2.0 * (z * z);

//     return  atan2(t3, t4);
// }

// int determine_x_coord( int x0, double robot_orient, double laser_angle, double laser_measure, double ratio ){
//     double line_angle = robot_orient + laser_angle;
//     return (int) (x0 + (laser_measure * cos(line_angle) ) / ratio);

// }

// int determine_y_coord( int y0, double robot_orient, double laser_angle, double laser_measure, double ratio ){
//     double line_angle = robot_orient + laser_angle;

//     return (int) (y0 + (laser_measure * sin(line_angle) ) / ratio);

// }



int main(int argc, char **argv) {

    // if( argc != 2){
    //     printf("ERROR: Missing meters/pixel ratio argument\n");
    //     exit(1);
    // }

    FILE *file = fopen("bag_data.txt", "r");
    //FILE *export = fopen("export.txt", "a");

    if(!file) return 1;
    //if(!export) return 2;

    node_t * head = NULL;
    head = (node_t *) malloc(sizeof(node_t));
    if (head == NULL) return 3;

    head->x = 0;
    head->y = 0;
    head->l_odd = 0;
    head->next = NULL;
    
    node_t *current = head;


    
    double meter_per_pixel = 0.05;
    // sscanf(argv[1], "%lf", &meter_per_pixel);

    double l_occ = log(0.70/0.30);
    double l_free = log(0.30/0.70);

    int x_max = 0, x_min = 0;
    int y_max = 0, y_min = 0;

    
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
                //fprintf(export, "%d %d %f\n", point_x, point_y, l_occ);
                node_t * new_node;
                new_node = (node_t *) malloc(sizeof(node_t));
                current->x = point_x;
                current->y = point_y;
                current->l_odd = l_occ;
                current->next = new_node;
                current = current->next;
                break;
            }
            node_t * new_node;
            new_node = (node_t *) malloc(sizeof(node_t));
            current->x = point_x;
            current->y = point_y;
            current->l_odd = l_free;
            current->next = new_node;
            current = current->next;
            //fprintf(export, "%d %d %f\n", point_x, point_y, l_free);

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
    current->next = NULL;

    //fclose(export);
    // fclose(file);

    // FILE *size = fopen("size.txt", "w");

    int size_x = x_max + abs(x_min) + 1;
    int size_y = y_max + abs(y_min) + 1;

    double **mat = (double **) malloc(size_x * sizeof(double*));
    for(int i = 0; i < size_x; i++) { 
        mat[i] = (double *) malloc(size_y * sizeof(double)); 
        for (int j = 0; j < size_y; j++)
        {
            mat[i][j] = 0;
        }
        
    }

    current = head;

    while (current->next != NULL)
    {
        mat[current->x + abs(x_min)][current->y + abs(y_min)] = mat[current->x + abs(x_min)][current->y + abs(y_min)] + current->l_odd;
        current = current->next;
    }
    
    // fprintf(size ,"%d %d %d %d", x_max, x_min, y_max, y_min);
    // fclose(size);
    FILE *export = fopen("export.txt", "a");

    for(int i = 0; i < size_x; i++){
        for (int j = 0; j < size_y; j++)
        {
            if(j == size_y - 1){
               fprintf(export, "%f\n", mat[i][j]);
               continue; 
            }
            fprintf(export, "%f,", mat[i][j]);
        }
    }

    fclose(export);
    return 0;
}

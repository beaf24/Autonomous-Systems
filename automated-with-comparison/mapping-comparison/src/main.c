#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>
#include "operations.h"
#include <math.h>
#include <limits.h>
#include <fcntl.h>
#include <string.h>


typedef struct Node
{
    int x;
    int y;
    double l_odd;
    struct Node* next;
    
}node_t;




int main(int argc, char **argv) {

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    char* dir;
    dir = dirname(strdup(cwd));
    printf("%s\n", dir);

    if( argc != 3){
        printf("ERROR: Missing meters/pixel ratio argument\n");
        exit(1);
    }

    char bagdata_path[PATH_MAX - 2];
    strcat(bagdata_path, dir);
    strcat(bagdata_path, "/mapping/data/bag_data.txt");

    FILE *file = fopen(bagdata_path, "r");

    if(!file) { printf("ERROR file"); return 1; }

    double meter_per_pixel;
    sscanf(argv[1], "%lf", &meter_per_pixel);

    double p_free;
    sscanf(argv[2], "%lf", &p_free);
    double p_occ = 1 - p_free;

    double l_free = log(p_free/p_occ);
    double l_occ = log(p_occ/p_free);

    node_t * head = NULL;
    head = (node_t *) malloc(sizeof(node_t));
    if (head == NULL) return 3;

    head->x = 0;
    head->y = 0;
    head->l_odd = 0;
    head->next = NULL;
    
    node_t *current = head;

    int x_max = 0, x_min = 0;
    int y_max = 0, y_min = 0;

    
    float x, y, quat_w, quat_z, angle, measure;
    double robot_orientation = 0;
    int robot_x = 0, robot_y = 0;



    // Read data
    while(fscanf(file, "%f %f %f %f %f %f ", &x, &y, &quat_w, &quat_z, &angle, &measure) == 6){

        if( (int) (measure/meter_per_pixel) == 0) continue;
        int obs_x = 0, obs_y = 0;
        int wall_x = 0, wall_y = 0;
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

        wall_x = determine_x_coord(obs_x, robot_orientation, angle, 0.2, meter_per_pixel);
        wall_y = determine_x_coord(obs_y, robot_orientation, angle, 0.2, meter_per_pixel);

        // Set new x_max or x_min
        if(obs_x > x_max) x_max = obs_x;
        else if( obs_x < x_min) x_min = obs_x;

        // Set new y_max or y_min
        if(obs_y > y_max) y_max = obs_y;
        else if( obs_y < y_min) y_min = obs_y;

        // Set new x_max or x_min
        if(wall_x > x_max) x_max = wall_x;
        else if( wall_x < x_min) x_min = wall_x;

        // Set new y_max or y_min
        if(wall_y > y_max) y_max = wall_y;
        else if( wall_y < y_min) y_min = wall_y;

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

        while(1){
            if (point_x == wall_x && point_y == wall_y){
                node_t * new_node;
                new_node = (node_t *) malloc(sizeof(node_t));
                current->x = point_x;
                current->y = point_y;
                current->l_odd = l_occ;
                current->next = new_node;
                current = current->next;
                // break;
            }

            node_t * new_node;
            new_node = (node_t *) malloc(sizeof(node_t));
            current->x = point_x;
            current->y = point_y;
            current->l_odd = l_occ;
            current->next = new_node;
            current = current->next;

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


    char export_path[PATH_MAX - 2];
    strcat(export_path, dir);
    strcat(export_path, "/mapping/data/export.txt");

    FILE *export = fopen(export_path, "a");
    if(!export) { printf("ERROR export"); return 1; }

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

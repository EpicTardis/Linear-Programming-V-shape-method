//
// Created by Ã∆“’∑Â on 2018/7/14.
//

#ifndef INC_2D_LINEAR_PROBLEM_MODELS_H
#define INC_2D_LINEAR_PROBLEM_MODELS_H

#include "floating_number_helper.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef struct line {
	// ax + by >= c
	double param_a;
	double param_b;
	double param_c;
	double slope_value;

	__host__ __device__
	line (){}

	__host__ __device__
	line(double _param_a, double _param_b, double _param_c):param_a(_param_a), param_b(_param_b), param_c(_param_c){}
	__host__ __device__
		line(double _param_a, double _param_b, double _param_c, double _slope_value) : param_a(_param_a), param_b(_param_b), param_c(_param_c), slope_value(_slope_value) {}
} line;


typedef struct point {
	// (x, y)
	double pos_x;
	double pos_y;
	__host__ __device__
	point (){}
	__host__ __device__
	point (double _pos_x,double _pos_y):pos_x(_pos_x),pos_y(_pos_y){}
	__host__ __device__
	bool operator==(const point& p) {
		return equals(this->pos_x, p.pos_x) && equals(this->pos_y, p.pos_y);
	}
} point;

typedef struct line_pair {
	line line1;
	line line2;
	bool line1_valid = TRUE;
	bool line2_valid = TRUE;
} line_pair;

typedef struct rotation_vector {
	double sine;
	double cosine;
} rotation_vector;

// the functions below may not be all used

line * generate_line_from_abc(double param_a, double param_b, double param_c); // ax + by = c
line * generate_line_from_kb(double k, double b); // y = kx + b
line * generate_line_from_2points(point * p1, point * p2); //

point * generate_point_from_xy(double pos_x, double pos_y);
point * generate_intersection_point(line * line1, line * line2);

double compute_slope(line * line);
int is_parallel(line * line1, line * line2);

#endif //INC_2D_LINEAR_PROBLEM_MODELS_H

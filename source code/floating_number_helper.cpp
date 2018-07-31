//
// Created by ÌÆÒÕ·å on 2018/7/14.
//

#include "floating_number_helper.h"

int equals(double num1, double num2) {
	return fabs(num1 - num2) < EPS ? TRUE : FALSE;
}

int strictly_larger(double num1, double num2) {
	return (num1 - EPS > num2) ? TRUE : FALSE;
}

int strictly_less(double num1, double num2) {
	return (num1 + EPS < num2) ? TRUE : FALSE;
}

double random_double_bounds(double left_bound, double right_bound) {
	float scale = rand() / (float)RAND_MAX;
	double modified_left = left_bound + EPS;
	double modified_right = right_bound - EPS;
	return modified_left + scale * (modified_right - modified_left);
}

double random_double() {
	return random_double_bounds(RANDOM_LEFT_BOUND, RANDOM_RIGHT_BOUND);
}
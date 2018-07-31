//
// Created by Ã∆“’∑Â on 2018/7/14.
//
#ifndef INC_2D_LINEAR_PROBLEM_FLOATING_NUMBER_HELPER_H
#define INC_2D_LINEAR_PROBLEM_FLOATING_NUMBER_HELPER_H

#include <math.h>
#include <float.h>
#include <stdlib.h>

#define TRUE    1
#define FALSE   0

#define EPS                 1e-5
#define RANDOM_LEFT_BOUND   (-10000)
#define RANDOM_RIGHT_BOUND  (+10000)


int equals(double num1, double num2);                               // num1 == num2
int strictly_larger(double num1, double num2);                      // num1 > num2
int strictly_less(double num1, double num2);                        // num1 < num2
double random_double_bounds(double left_bound, double right_bound); // (left_bound, right_bound)
double random_double();                                             //

#endif //INC_2D_LINEAR_PROBLEM_FLOATING_NUMBER_HELPER_H
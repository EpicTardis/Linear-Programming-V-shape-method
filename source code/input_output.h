//
// Created by Ã∆“’∑Â on 2018/7/14.
//
#define _CRT_SECURE_NO_WARNINGS

#ifndef INC_2D_LINEAR_PROBLEM_INPUT_OUTPUT_H
#define INC_2D_LINEAR_PROBLEM_INPUT_OUTPUT_H

#include <stdio.h>
#include "models.h"

#define DEFAULT_ANS_LEN 10000

typedef struct inputs {
	int number;                     // how many lines we have
	//line ** lines;                  // lines stored as pointer array
	std::vector<line> lines;
	double obj_function_param_a;    // parameter a of objective function
	double obj_function_param_b;    // parameter b of objective function
} inputs;

typedef struct answer {
	double answer_b;                // answer
	line * line1;                   // which line
	line * line2;                   //
	point * intersection_point;     // the intersection point
} answer;

inputs * read_from_file(char * filename);
char * generate_ans_string(answer * ans);
void free_inputs(inputs ** input);
void free_ans(answer ** ans);

#endif //INC_2D_LINEAR_PROBLEM_INPUT_OUTPUT_H
//
// Created by Ã∆“’∑Â on 2018/7/14.
//

#include "input_output.h"

inputs * read_from_file(char * filename) {
	inputs* input = new inputs;
	FILE * input_file = fopen(filename, "r");
	if (input_file == NULL) {
		printf("Cannot open the file.");
		exit(1);
	}
	int num_of_lines;
	fscanf(input_file, "%d", &num_of_lines);
	//line ** lines = (line **)malloc(sizeof(line *) * num_of_lines);
	std::vector<line> lines(num_of_lines);
	for (int line_no = 0; line_no < num_of_lines; line_no++) {
		double param_a, param_b, param_c;
		fscanf(input_file, "%lf %lf %lf", &param_a, &param_b, &param_c);
		//line * new_line = generate_line_from_abc(param_a, param_b, param_c);
		lines[line_no] = line(param_a, param_b, param_c);
	}
	fscanf(input_file, "%lf %lf", &input->obj_function_param_a, &input->obj_function_param_b);
	input->lines.assign(lines.begin(),lines.end());
	input->number = num_of_lines;
	fclose(input_file);
	return input;
}

char * generate_ans_string(answer * ans) {
	char * ans_string = (char *)malloc(sizeof(char) * DEFAULT_ANS_LEN);
	sprintf(ans_string, "Answer is: %lf\n"
		"Line 1 is : %lfx + %lfy >= %lf\n"
		"Line 2 is : %lfx + %lfy >= %lf\n"
		"Intersection point is (%lf, %lf)\n",
		ans->answer_b,
		ans->line1->param_a, ans->line1->param_b, ans->line1->param_c,
		ans->line2->param_a, ans->line2->param_b, ans->line2->param_c,
		ans->intersection_point->pos_x, ans->intersection_point->pos_y);
	return ans_string;
}

void free_inputs(inputs ** input) {
	inputs * aim = *input;
	/*for (int i = 0; i < aim->number; i++) {
		free(aim->lines[i]);
	}*/
	//free(aim->lines);
	free(aim);
}

void free_ans(answer ** ans) {
	answer * aim = *ans;
	//    free(aim->line1);
	//    free(aim->line2);
	//    take care of double-free!
	free(aim->intersection_point);
	free(aim);
}
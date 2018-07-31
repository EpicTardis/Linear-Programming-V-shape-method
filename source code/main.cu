#include "floating_number_helper.h"
#include "input_output.h"
#include <vector>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>


//
// this function is what you need to finish
// @Usage : to solve the problem
// @Input : input containing all data needed
// @Output: answer containing all necessary data
//  you can find the definition of the two structs above in
//      input_output.h
//


//Second GPU version of 2D linear problem
//By Dong Yiming, Jul 26, 2018

#define MAXBOUND 1e20   //Initial left and right bound
#define MIN_NUMBER_OF_LINES 1000
#define EMPTYPOINT point(NAN,NAN)
#define EMPTYLINE line(0,0,0)

int begintime[100], endtime[100], duringtime[100];

//Only functions defined in .cu could be used, so I redefined them

__host__ __device__
inline int Strictly_less(double num1, double num2) {
	return (num1 + EPS < num2) ? TRUE : FALSE;
}

__host__ __device__
inline int Strictly_larger(double num1, double num2) {
	return (num1 - EPS > num2) ? TRUE : FALSE;
}

__host__ __device__
inline int Equals(double num1, double num2) {
	return fabs(num1 - num2) < EPS ? TRUE : FALSE;
}

__host__ __device__
double Compute_slope(line l) {
	if (Equals(l.param_b, 0)) {
		if (l.param_a > 0) {
			return -FLT_MAX;
		}
		return FLT_MAX;
	}
	return -l.param_a / l.param_b;
}

__host__ __device__
inline int Is_parallel(line line1, line line2) {
	return Equals(line1.param_a * line2.param_b, line1.param_b * line2.param_a);
}

__host__ __device__
point Generate_intersection_point(line line1, line line2) {
	point new_point;
	if (Is_parallel(line1, line2)) {
		return EMPTYPOINT;
	}
	else {
		new_point.pos_x = (line1.param_c * line2.param_b - line1.param_b * line2.param_c)
			/ (line1.param_a * line2.param_b - line1.param_b * line2.param_a);
		new_point.pos_y = (line1.param_c * line2.param_a - line1.param_a * line2.param_c)
			/ (line1.param_b * line2.param_a - line1.param_a * line2.param_b);
	}
	return new_point;
}

__host__ __device__
point Generate_intersection_point(line l, double boundary) {
	point new_point;
	if (Equals(l.param_b, 0)) {
		return EMPTYPOINT;
	}
	else {
		new_point.pos_x = boundary;
		new_point.pos_y = (l.param_c - l.param_a*boundary) / l.param_b;
	}
	return new_point;
}

struct compare_boundary_value {
	double boundary;
	compare_boundary_value(double _boundary) :boundary(_boundary) {}
	__host__ __device__
		bool operator()(line line1, line line2) {
		return Strictly_less(Generate_intersection_point(line1, boundary).pos_y, Generate_intersection_point(line2, boundary).pos_y);
	}
};

//Rotation and inverse-rotation functions
__host__ __device__
point rotation(point p, rotation_vector rv) {
	point rotatedpoint;
	rotatedpoint.pos_x = p.pos_x*rv.cosine - p.pos_y*rv.sine;
	rotatedpoint.pos_y = p.pos_x*rv.sine + p.pos_y*rv.cosine;
	return rotatedpoint;
}

__host__ __device__
line rotation(line l, rotation_vector rv) {
	line rotatedline;
	rotatedline.param_a = l.param_a*rv.cosine - l.param_b*rv.sine;
	rotatedline.param_b = l.param_a*rv.sine + l.param_b*rv.cosine;
	rotatedline.param_c = l.param_c;
	rotatedline.slope_value = Compute_slope(rotatedline);
	return rotatedline;
}

__host__ __device__
point inverse_rotation(point p, rotation_vector rv) {
	rv.sine = -rv.sine;
	return rotation(p, rv);
}

__host__ __device__
line inverse_rotation(line l, rotation_vector rv) {
	rv.sine = -rv.sine;
	return rotation(l, rv);
}

//Unary function of line rotation
struct rotate_lines {
	rotation_vector rv;
	rotate_lines(rotation_vector _rv) :rv(_rv) {}
	__host__ __device__
		line operator()(line paraline) {
		return rotation(paraline, rv);
	}
};

//Unary function of classfying lines into I+,I- or I0
struct in_Ipos {
	__host__ __device__
		bool operator()(line paraline) {
		return Strictly_larger(paraline.param_b, 0);
	}
};

struct in_Ineg {
	__host__ __device__
		bool operator()(line paraline) {
		return Strictly_less(paraline.param_b, 0);
	}
};

struct in_I0_left {
	__host__ __device__
		bool operator()(line paraline) {
		return Equals(paraline.param_b, 0) && Strictly_less(paraline.param_a, 0);
	}
};

struct in_I0_right {
	__host__ __device__
		bool operator()(line paraline) {
		return Equals(paraline.param_b, 0) && Strictly_larger(paraline.param_a, 0);
	}
};

struct line_to_boundary {
	__host__ __device__
		double operator()(line l) {
		return l.param_c / l.param_a;
	}
};

//Binary function of giving the criterion of comparing two lines to judge the new boundary
struct compare_line_boundary {
	__host__ __device__
		bool operator()(line line1, line line2) {
		return Strictly_less(line1.param_c / line1.param_a, line2.param_c / line2.param_a);
	}
};

//Determining whether the line is useful or redundant
struct useful_Ipos {
	point intersection_point;
	useful_Ipos(point _intersection_point) :intersection_point(_intersection_point) {}
	__host__ __device__
		bool operator()(line paraline) {
		return !Strictly_less(Generate_intersection_point(paraline, intersection_point.pos_x).pos_y, intersection_point.pos_y);
	}
};

struct useful_Ineg {
	point intersection_point;
	useful_Ineg(point _intersection_point) :intersection_point(_intersection_point) {}
	__host__ __device__
		bool operator()(line paraline) {
		return !Strictly_larger(Generate_intersection_point(paraline, intersection_point.pos_x).pos_y, intersection_point.pos_y);
	}
};



answer * compute(inputs * input) {

	int num = input->number;
	double objective_function_value;

	//All of the memory of object "ans" is allocated at this
	//And I don't use pointer to assign pointer, which may cause double free
	answer * ans = (answer *)malloc(sizeof(answer));
	ans->answer_b = DBL_MAX;
	ans->intersection_point = (point*)malloc(sizeof(point));
	ans->line1 = (line*)malloc(sizeof(line));
	ans->line2 = (line*)malloc(sizeof(line));

	//All lines transferred from *lines[] to vector

	thrust::host_vector <line>all_lines_host = input->lines;
	thrust::device_vector <line>all_lines_device = all_lines_host;

	//1.Rotate coordinary system
	//1.1 Construct the rotation vector

	rotation_vector rv;
	if (Equals(input->obj_function_param_a, 0)) {
		rv.cosine = 1; rv.sine = 0;
	}
	else if (Equals(input->obj_function_param_b, 0)) {
		rv.sine = 1; rv.cosine = 0;
	}
	else {
		rv.sine = input->obj_function_param_a / sqrt(input->obj_function_param_a*input->obj_function_param_a + input->obj_function_param_b*input->obj_function_param_b);
		rv.cosine = input->obj_function_param_b / sqrt(input->obj_function_param_a*input->obj_function_param_a + input->obj_function_param_b*input->obj_function_param_b);
	}

	//1.2 Rotate all the lines

	//thrust::device_vector <line>all_lines_device(num);
	thrust::transform(all_lines_device.begin(), all_lines_device.end(),
		all_lines_device.begin(), rotate_lines(rv));

	//1.3 Modify the objective function
	double rotated_obj_function_param_b = sqrt(input->obj_function_param_a*input->obj_function_param_a + input->obj_function_param_b*input->obj_function_param_b);
	//rotated_obj_function_param_a == 0


	//2.Classify lines into I+,I- and I0

	thrust::device_vector <line>rotated_lines_device_Ipos(num);
	thrust::device_vector <line>rotated_lines_device_Ineg(num);
	thrust::device_vector <line>rotated_lines_device_I0_left(num);
	thrust::device_vector <line>rotated_lines_device_I0_right(num);

	thrust::device_vector <line>::iterator Ipos_end = thrust::copy_if(all_lines_device.begin(), all_lines_device.end(), rotated_lines_device_Ipos.begin(), in_Ipos());
	thrust::device_vector <line>::iterator Ineg_end = thrust::copy_if(all_lines_device.begin(), all_lines_device.end(), rotated_lines_device_Ineg.begin(), in_Ineg());
	thrust::device_vector <line>::iterator I0_left_end = thrust::copy_if(all_lines_device.begin(), all_lines_device.end(), rotated_lines_device_I0_left.begin(), in_I0_left());
	thrust::device_vector <line>::iterator I0_right_end = thrust::copy_if(all_lines_device.begin(), all_lines_device.end(), rotated_lines_device_I0_right.begin(), in_I0_right());

	rotated_lines_device_Ipos.resize((Ipos_end - rotated_lines_device_Ipos.begin()));
	rotated_lines_device_Ineg.resize((Ineg_end - rotated_lines_device_Ineg.begin()));
	rotated_lines_device_I0_left.resize((I0_left_end - rotated_lines_device_I0_left.begin()));
	rotated_lines_device_I0_right.resize((I0_right_end - rotated_lines_device_I0_right.begin()));

	//3.Take the test line and remove lines
	//3.1 Initialize left and right boundary

	double left_boundary = -MAXBOUND, right_boundary = MAXBOUND;

	//Use I0_left to update left boundary
	if (!rotated_lines_device_I0_left.empty()) {
		thrust::device_vector <line>::iterator max_left_boundary_iter = thrust::max_element(rotated_lines_device_I0_left.begin(), rotated_lines_device_I0_left.end(), compare_line_boundary());
		thrust::host_vector<line> max_left_boundary_vector(max_left_boundary_iter, max_left_boundary_iter + 1);
		left_boundary = max_left_boundary_vector[0].param_c / max_left_boundary_vector[0].param_a;
	}

	//Use I0_right to update right boundary
	if (!rotated_lines_device_I0_right.empty()) {
		thrust::device_vector <line>::iterator min_right_boundary_iter = thrust::min_element(rotated_lines_device_I0_right.begin(), rotated_lines_device_I0_right.end(), compare_line_boundary());
		thrust::host_vector<line> min_right_boundary_vector(min_right_boundary_iter, min_right_boundary_iter + 1);
		right_boundary = min_right_boundary_vector[0].param_c / min_right_boundary_vector[0].param_a;
	}

	//Remove redundant lines: keep removing if the number of lines is not smaller than a limiteded size

	while (rotated_lines_device_Ipos.size() + rotated_lines_device_Ineg.size() >= MIN_NUMBER_OF_LINES) {
		//3.2 Determine the line in I+ and I- with largest/smallest y-axis of intersection point with boundaries respectively


		//Calculate the max boundary lines in I+ and I- and get the intersection point
		line max_left_boundary_line = *thrust::max_element(rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end(), compare_boundary_value(left_boundary));
		line max_right_boundary_line = *thrust::max_element(rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end(), compare_boundary_value(right_boundary));
		point intersection_point_Ipos = Generate_intersection_point(max_left_boundary_line, max_right_boundary_line);


		line min_left_boundary_line = *thrust::min_element(rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end(), compare_boundary_value(left_boundary));
		line min_right_boundary_line = *thrust::min_element(rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end(), compare_boundary_value(right_boundary));
		point intersection_point_Ineg = Generate_intersection_point(min_left_boundary_line, min_right_boundary_line);

		//3.3 Generate test line
		double test_line_Ipos, test_line_Ineg;
		if (intersection_point_Ipos == EMPTYPOINT)
			test_line_Ipos = NAN;
		else test_line_Ipos = intersection_point_Ipos.pos_x;
		if (intersection_point_Ineg == EMPTYPOINT)
			test_line_Ineg = NAN;
		else test_line_Ineg = intersection_point_Ineg.pos_x;

		//3.4 Remove redundant lines

		thrust::device_vector <line>::iterator Ipos_useful_end = thrust::copy_if(rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end(), rotated_lines_device_Ipos.begin(), useful_Ipos(intersection_point_Ipos));
		thrust::device_vector <line>::iterator Ineg_useful_end = thrust::copy_if(rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end(), rotated_lines_device_Ineg.begin(), useful_Ineg(intersection_point_Ineg));
		rotated_lines_device_Ipos.resize(Ipos_useful_end - rotated_lines_device_Ipos.begin());
		rotated_lines_device_Ineg.resize(Ineg_useful_end - rotated_lines_device_Ineg.begin());

		//3.5 Mark the test line as the new boundary
		line max_Ipos_line_of_Ipos_testline = *thrust::max_element(rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end(), compare_boundary_value(test_line_Ipos));
		line min_Ineg_line_of_Ipos_testline = *thrust::min_element(rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end(), compare_boundary_value(test_line_Ipos));
		line max_Ipos_line_of_Ineg_testline = *thrust::max_element(rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end(), compare_boundary_value(test_line_Ineg));
		line min_Ineg_line_of_Ineg_testline = *thrust::min_element(rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end(), compare_boundary_value(test_line_Ineg));

		//The direction of optimal solution can be decided as follows by the slope of line in I+:
		//if slope<0 then the optimal solution in on the right, thus the test line maybe the left boundary
		//if slope>0 then the optimal solution in on the left, thus the test line maybe the right boundary

		//After knowing the direction, compare the test line and current boundary to decide whether we modify the boundary
		if (Strictly_less(Generate_intersection_point(max_Ipos_line_of_Ipos_testline, test_line_Ipos).pos_y, Generate_intersection_point(min_Ineg_line_of_Ipos_testline, test_line_Ipos).pos_y)) {
			if (Strictly_larger(max_Ipos_line_of_Ipos_testline.slope_value, 0) && Strictly_less(test_line_Ipos, right_boundary)) {
				right_boundary = test_line_Ipos;
			}
			else if (Strictly_less(max_Ipos_line_of_Ipos_testline.slope_value, 0) && Strictly_larger(test_line_Ipos, left_boundary)) {
				left_boundary = test_line_Ipos;
			}
		}
		else {
			if (Strictly_larger(max_Ipos_line_of_Ipos_testline.slope_value, min_Ineg_line_of_Ipos_testline.slope_value) && Strictly_less(test_line_Ipos, right_boundary))
				right_boundary = test_line_Ipos;
			else if (Strictly_less(max_Ipos_line_of_Ipos_testline.slope_value, min_Ineg_line_of_Ipos_testline.slope_value) && Strictly_larger(test_line_Ipos, left_boundary))
				left_boundary = test_line_Ipos;
		}


		if (Strictly_less(Generate_intersection_point(max_Ipos_line_of_Ineg_testline, test_line_Ineg).pos_y, Generate_intersection_point(min_Ineg_line_of_Ineg_testline, test_line_Ineg).pos_y)) {
			if (Strictly_larger(max_Ipos_line_of_Ineg_testline.slope_value, 0) && Strictly_less(test_line_Ineg, right_boundary)) {
				right_boundary = test_line_Ineg;
			}
			else if (Strictly_less(max_Ipos_line_of_Ineg_testline.slope_value, 0) && Strictly_larger(test_line_Ineg, left_boundary)) {
				left_boundary = test_line_Ineg;
			}
		}
		else {
			if (Strictly_larger(max_Ipos_line_of_Ineg_testline.slope_value, min_Ineg_line_of_Ineg_testline.slope_value) && Strictly_less(test_line_Ineg, right_boundary))
				right_boundary = test_line_Ineg;
			else if (Strictly_less(max_Ipos_line_of_Ineg_testline.slope_value, min_Ineg_line_of_Ineg_testline.slope_value) && Strictly_larger(test_line_Ineg, left_boundary))
				left_boundary = test_line_Ineg;
		}
	}

	//After the circulation, now get the useful lines
	std::vector<line> useful_lines;
	useful_lines.insert(useful_lines.end(), rotated_lines_device_Ipos.begin(), rotated_lines_device_Ipos.end());
	useful_lines.insert(useful_lines.end(), rotated_lines_device_Ineg.begin(), rotated_lines_device_Ineg.end());

	//4.Use CPU to solve the small problem which involves less than 10 lines
	for (int i = 0; i < useful_lines.size(); i++)
		for (int j = i + 1; j < useful_lines.size(); j++) {
			bool flag = TRUE;
			point new_point = Generate_intersection_point(useful_lines[i], useful_lines[j]);
			if (!(new_point == EMPTYPOINT)) {
				for (int k = 0; k < useful_lines.size(); k++) {
					if (Strictly_less(useful_lines[k].param_a*new_point.pos_x + useful_lines[k].param_b*new_point.pos_y, useful_lines[k].param_c))
						flag = FALSE;
				}
				objective_function_value = new_point.pos_y*sqrt(input->obj_function_param_a*input->obj_function_param_a + input->obj_function_param_b*input->obj_function_param_b);
				if (strictly_less(objective_function_value, ans->answer_b) && flag) {
					//5.Rotate back the lines and point if the objective function value of new_point is less than current value
					ans->answer_b = objective_function_value;
					*ans->line1 = inverse_rotation(line(useful_lines[i].param_a, useful_lines[i].param_b, useful_lines[i].param_c, useful_lines[i].slope_value), rv);
					*ans->line2 = inverse_rotation(line(useful_lines[j].param_a, useful_lines[j].param_b, useful_lines[j].param_c, useful_lines[j].slope_value), rv);
					*ans->intersection_point = inverse_rotation(new_point, rv);
				}
			}
		}
	return ans;
}


int main() {
	int inputfilename[] = { 15000000};
	FILE* output_file = fopen("../result.dat", "w");
	for (auto i : inputfilename) {
		// 1. get the input data
		//inputs * input = read_from_file("../1000000_0.dat");
		char a[50] = "../"; char b[50]; itoa(i, b, 10); char* c = "_0.dat"; strcat(b, c); strcat(a, b);
		inputs * input = read_from_file(a);
		// 2. get the answer
		answer * ans = compute(input);
		for (int i = 0; i < 100; i++) {
			begintime[i] = clock();
			ans = compute(input);
			endtime[i] = clock();
		}
		// 3. display result and free memory
		char * ans_string = generate_ans_string(ans);
		printf("%s", ans_string);
		for (int i = 0; i < 100; i++) {
			duringtime[i] = endtime[i] - begintime[i];
			fprintf(output_file, "%d\t", duringtime[i]);
			printf("time:%d\t", duringtime[i]);
		}
		printf("\n\n");
	
		free_inputs(&input);
		free_ans(&ans);
		free(ans_string);
	}	
	fclose(output_file);
	return 0;
}

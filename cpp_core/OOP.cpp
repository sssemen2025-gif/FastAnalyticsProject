#include <vector>
#include <cmath>
#include <iostream>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;

vector<double>skol_sr(vector<double>& data, size_t R)
{
	vector<double>result;
	result.reserve(data.size() - R + 1);
	size_t s = data.size() - R + 1;
	double a = 0;
	for (size_t i = 0; i < s; ++i)
	{
		for (size_t j = 0; j < R; ++j) {
			a += data[j + i];


		}
		a = a / R;
		result.push_back(a);
		a = 0;
	}
	return result;
}
void bubble_sort_1(vector<double>& data)
{
	int a = data.size();
	int flag = 1;
	while (flag == 1)
	{
		flag = 0;
		
		for (int i = 0; i < data.size() - 1; ++i)
		{
			
			if (data[i] > data[i + 1])
			{
				swap(data[i], data[i + 1]);
				flag = 1;
			}

		}
	}
	
}
void bubble_sort_2(vector<double>& arr)  // по ссылке, меняет оригинал
{
	int n = arr.size();
	for (int i = 0; i < n - 1; ++i)           // внешний цикл: количество проходов
	{
		for (int j = 0; j < n - i - 1; ++j)   // внутренний цикл: сравнение пар
		{
			if (arr[j] > arr[j + 1])
			{
				swap(arr[j], arr[j + 1]);
			}
		}
	}
}




	
	

PYBIND11_MODULE(sorting, m) {
	m.def("sort_1", &bubble_sort_1, py::arg("data"));
	m.def("sort_2", &bubble_sort_2, py::arg("data"));
}

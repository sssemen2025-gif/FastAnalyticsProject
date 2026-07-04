#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

std::vector<double> moving_average(const std::vector<double>& data, size_t window_size) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (window_size == 0 || window_size > data.size()) {
        throw std::invalid_argument("Invalid window size");
    }
    
    std::vector<double> result;
    result.reserve(data.size() - window_size + 1);
    
    double window_sum = 0.0;
    for (size_t i = 0; i < window_size; ++i) {
        window_sum += data[i];
    }
    result.push_back(window_sum / window_size);
    
    for (size_t i = window_size; i < data.size(); ++i) {
        window_sum = window_sum - data[i - window_size] + data[i];
        result.push_back(window_sum / window_size);
    }
    
    return result;
}

std::vector<double> ewma(const std::vector<double>& data, double alpha = 0.1) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("Alpha must be in (0, 1]");
    }
    
    std::vector<double> result;
    result.reserve(data.size());
    
    double current = data[0];
    result.push_back(current);
    
    for (size_t i = 1; i < data.size(); ++i) {
        current = alpha * data[i] + (1 - alpha) * current;
        result.push_back(current);
    }
    
    return result;
}

std::vector<double> zscore(const std::vector<double>& data) {
    if (data.size() < 2) {
        throw std::invalid_argument("Need at least 2 values");
    }
    
    double sum = 0.0;
    for (double val : data) sum += val;
    double mean = sum / data.size();
    
    double variance_sum = 0.0;
    for (double val : data) {
        double diff = val - mean;
        variance_sum += diff * diff;
    }
    double stddev = std::sqrt(variance_sum / data.size());
    
    std::vector<double> z_scores;
    z_scores.reserve(data.size());
    
    if (stddev < 1e-10) {
        for (size_t i = 0; i < data.size(); ++i) {
            z_scores.push_back(0.0);
        }
    } else {
        for (double val : data) {
            z_scores.push_back((val - mean) / stddev);
        }
    }
    
    return z_scores;
}

std::vector<size_t> find_anomalies(const std::vector<double>& data, double threshold = 3.0) {
    std::vector<double> z_scores = zscore(data);
    std::vector<size_t> anomalies;
    anomalies.reserve(data.size() / 100);
    
    for (size_t i = 0; i < z_scores.size(); ++i) {
        if (std::abs(z_scores[i]) > threshold) {
            anomalies.push_back(i);
        }
    }
    return anomalies;
}

double median(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        return sorted[n/2];
    }
}



// 6. Линейная регрессия (OLS)
// Возвращает: (intercept, slope, r_squared)
std::tuple<double, double, double> linear_regression(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.empty()) throw std::invalid_argument("Input vectors cannot be empty");
    if (x.size() != y.size()) throw std::invalid_argument("x and y must have the same size");

    size_t n = x.size();
    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    double denom = n * sumX2 - sumX * sumX;
    if (std::abs(denom) < 1e-12)
        throw std::runtime_error("Cannot compute linear regression: denominator too small");

    double slope = (n * sumXY - sumX * sumY) / denom;
    double intercept = (sumY - slope * sumX) / n;

    // R-squared
    double ss_res = 0.0, ss_tot = 0.0;
    double meanY = sumY / n;
    for (size_t i = 0; i < n; ++i) {
        double pred = intercept + slope * x[i];
        ss_res += (y[i] - pred) * (y[i] - pred);
        ss_tot += (y[i] - meanY) * (y[i] - meanY);
    }
    double r_squared = 1.0 - ss_res / ss_tot;
    return std::make_tuple(intercept, slope, r_squared);
}

// 7. Метрики качества (MAE, MSE, RMSE, R²)
// Возвращает: (mae, mse, rmse, r2)
std::tuple<double, double, double, double> regression_metrics(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty()) throw std::invalid_argument("Input vectors cannot be empty");
    if (y_true.size() != y_pred.size()) throw std::invalid_argument("Vectors must have the same size");

    size_t n = y_true.size();
    double mae = 0.0, mse = 0.0, sum_true = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = y_true[i] - y_pred[i];
        mae += std::abs(diff);
        mse += diff * diff;
        sum_true += y_true[i];
    }
    mae /= n;
    mse /= n;
    double rmse = std::sqrt(mse);

    // R²
    double mean_true = sum_true / n;
    double ss_res = mse * n;   // уже посчитано
    double ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ss_tot += (y_true[i] - mean_true) * (y_true[i] - mean_true);
    }
    double r2 = 1.0 - ss_res / ss_tot;
    return std::make_tuple(mae, mse, rmse, r2);
}

double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty()) throw std::invalid_argument("Input vectors cannot be empty");
    if (y_true.size() != y_pred.size()) throw std::invalid_argument("Vectors must have the same size");

    size_t n = y_true.size();
    double sum_true = 0.0, ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = y_true[i] - y_pred[i];
        ss_res += diff * diff;
        sum_true += y_true[i];
    }
    double mean_true = sum_true / n;
    for (size_t i = 0; i < n; ++i) {
        ss_tot += (y_true[i] - mean_true) * (y_true[i] - mean_true);
    }
    return 1.0 - ss_res / ss_tot;
}

// 8. Поиск пиков (peak detection)
std::vector<size_t> find_peaks(const std::vector<double>& data, size_t lookahead = 1, double min_height = 0.0) {
    if (data.size() < 2) return {};
    std::vector<size_t> peaks;
    size_t n = data.size();
    for (size_t i = 0; i < n; ++i) {
        bool is_peak = true;
        // проверка локального максимума в окрестности ±lookahead
        size_t start = (i >= lookahead) ? i - lookahead : 0;
        size_t end = std::min(i + lookahead, n - 1);
        for (size_t j = start; j <= end; ++j) {
            if (j == i) continue;
            if (data[j] >= data[i]) {
                is_peak = false;
                break;
            }
        }
        if (!is_peak) continue;

        // проверка минимального превышения над соседями
        bool high_enough = true;
        for (size_t j = start; j <= end; ++j) {
            if (j == i) continue;
            if (data[i] - data[j] < min_height) {
                high_enough = false;
                break;
            }
        }
        if (high_enough)
            peaks.push_back(i);
    }
    return peaks;
}

PYBIND11_MODULE(fast_algo, m) {
    m.doc() = "High-performance analytics algorithms on C++";
    m.def("moving_average", &moving_average,
        "Calculate moving average with given window size",
        py::arg("data"), py::arg("window_size"));
    m.def("ewma", &ewma,
        "Exponential weighted moving average",
        py::arg("data"), py::arg("alpha") = 0.1);
    m.def("zscore", &zscore,
        "Calculate z-scores for anomaly detection",
        py::arg("data"));
    m.def("find_anomalies", &find_anomalies,
        "Find indices where z-score exceeds threshold",
        py::arg("data"), py::arg("threshold") = 3.0);
    m.def("median", &median,
        "Calculate median (robust to outliers)",
        py::arg("data"));

    // ========== Новые функции ==========
    m.def("linear_regression", &linear_regression,
        "Simple Linear Regression (OLS). Returns (intercept, slope, r_squared)",
        py::arg("x"), py::arg("y"));

    m.def("regression_metrics", &regression_metrics,
        "Calculate MAE, MSE, RMSE, R². Returns (mae, mse, rmse, r2)",
        py::arg("y_true"), py::arg("y_pred"));

    m.def("r2_score", &r2_score,
        "Calculate R² score",
        py::arg("y_true"), py::arg("y_pred"));

    m.def("find_peaks", &find_peaks,
        "Find peaks in 1D array. Returns indices of local maxima.",
        py::arg("data"), py::arg("lookahead") = 1, py::arg("min_height") = 0.0);

    m.attr("__version__") = "1.0.0";
}

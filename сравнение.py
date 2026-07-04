"""
Глобальный бенчмарк + валидация модуля fast_algo.
Сравнивает C++ реализации с чистым Python и известными библиотеками.
Размеры: 100k, 500k, 3M точек.
Выводит логи корректности и таблицы для README.md
"""

import time
import math
import random
import statistics as py_stats
import fast_algo
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sk_r2
import statsmodels.api as sm

# ================== Вспомогательные функции ==================
def measure(func, data, args=(), n_runs=5, warmup=2):
    for _ in range(warmup):
        func(data, *args)
    start = time.perf_counter()
    for _ in range(n_runs):
        func(data, *args)
    end = time.perf_counter()
    return (end - start) / n_runs

# ================== Чистые Python-реализации ==================
def py_moving_average(data, window):
    n = len(data)
    if window <= 0 or n < window:
        return []
    res = [0.0] * (n - window + 1)
    s = sum(data[:window])
    res[0] = s / window
    for i in range(window, n):
        s += data[i] - data[i - window]
        res[i - window + 1] = s / window
    return res

def py_ewma(data, alpha):
    if not data:
        return []
    res = [0.0] * len(data)
    res[0] = data[0]
    for i in range(1, len(data)):
        res[i] = alpha * data[i] + (1.0 - alpha) * res[i-1]
    return res

def py_median(data):
    if not data:
        raise ValueError("empty")
    return py_stats.median(data)

def py_zscore(data):
    if len(data) < 2:
        raise ValueError("Need at least 2 values")
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)
    if std < 1e-10:
        return [0.0] * n
    return [(x - mean) / std for x in data]

def py_find_anomalies(data, threshold=3.0):
    z = py_zscore(data)
    return [i for i, zv in enumerate(z) if abs(zv) > threshold]

def py_linear_regression(x, y):
    n = len(x)
    sumX = sum(x)
    sumY = sum(y)
    sumXY = sum(xi*yi for xi, yi in zip(x, y))
    sumX2 = sum(xi*xi for xi in x)
    denom = n * sumX2 - sumX * sumX
    slope = (n * sumXY - sumX * sumY) / denom
    intercept = (sumY - slope * sumX) / n
    y_pred = [intercept + slope * xi for xi in x]
    ss_res = sum((yi - yp)**2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - sumY/n)**2 for yi in y)
    r2 = 1 - ss_res / ss_tot
    return intercept, slope, r2

def py_regression_metrics(y_true, y_pred):
    n = len(y_true)
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p)**2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    ss_res = mse * n
    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true)**2 for t in y_true)
    r2 = 1 - ss_res / ss_tot
    return mae, mse, rmse, r2

def py_r2_score(y_true, y_pred):
    return py_regression_metrics(y_true, y_pred)[3]

def py_find_peaks(data, lookahead=1, min_height=0.0):
    peaks = []
    n = len(data)
    for i in range(n):
        is_peak = True
        start = max(0, i - lookahead)
        end = min(n, i + lookahead + 1)
        for j in range(start, end):
            if j == i: continue
            if data[j] >= data[i]:
                is_peak = False
                break
        if not is_peak:
            continue
        high_enough = True
        for j in range(start, end):
            if j == i: continue
            if data[i] - data[j] < min_height:
                high_enough = False
                break
        if high_enough:
            peaks.append(i)
    return peaks

# ================== Генерация данных ==================
np.random.seed(42)
random.seed(42)

sizes = {
    'Small (100k)': 100_000,
    'Medium (500k)': 500_000,
    'Large (3M)': 3_000_000
}

datasets = {}
for label, size in sizes.items():
    datasets[label] = np.random.uniform(0, 100, size).tolist()

x_reg_small = np.random.uniform(0, 10, 100_000).tolist()
y_reg_small = [2.5 + 1.8 * xi + random.gauss(0, 1) for xi in x_reg_small]
x_reg_med = np.random.uniform(0, 10, 500_000).tolist()
y_reg_med = [2.5 + 1.8 * xi + random.gauss(0, 1) for xi in x_reg_med]
x_reg_large = np.random.uniform(0, 10, 3_000_000).tolist()
y_reg_large = [2.5 + 1.8 * xi + random.gauss(0, 1) for xi in x_reg_large]

reg_data = {
    'Small (100k)': (x_reg_small, y_reg_small),
    'Medium (500k)': (x_reg_med, y_reg_med),
    'Large (3M)': (x_reg_large, y_reg_large)
}

metrics_data = {}
for label, (x, y_true) in reg_data.items():
    y_pred = [2.5 + 1.8 * xi for xi in x]
    metrics_data[label] = (y_true, y_pred)

# ================== Параметры функций ==================
WINDOW = 20
ALPHA = 0.3
THRESHOLD = 2.5
LOOKAHEAD = 5
MIN_HEIGHT = 1.0

# ================== Проверка корректности (короткий тест) ==================
def check_correctness():
    print("Быстрая проверка корректности на маленьком массиве...")
    test = [random.random()*100 for _ in range(1000)]
    x_test = [random.random()*10 for _ in range(1000)]
    y_test = [2.5 + 1.8*xi + random.gauss(0,1) for xi in x_test]
    y_pred_test = [2.5 + 1.8*xi for xi in x_test]

    cpp = fast_algo.moving_average(test, WINDOW)
    assert np.allclose(cpp, py_moving_average(test, WINDOW)), "moving_average fail"

    cpp = fast_algo.ewma(test, ALPHA)
    assert np.allclose(cpp, py_ewma(test, ALPHA)), "ewma fail"

    cpp = fast_algo.median(test)
    assert abs(cpp - py_median(test)) < 1e-9, "median fail"

    cpp = fast_algo.zscore(test)
    assert np.allclose(cpp, py_zscore(test)), "zscore fail"

    cpp = fast_algo.find_anomalies(test, THRESHOLD)
    assert cpp == py_find_anomalies(test, THRESHOLD), "find_anomalies fail"

    cpp = fast_algo.linear_regression(x_test, y_test)
    py = py_linear_regression(x_test, y_test)
    assert np.allclose(cpp[0], py[0]) and np.allclose(cpp[1], py[1]), "linear_regression fail"

    cpp = fast_algo.regression_metrics(y_test, y_pred_test)
    py = py_regression_metrics(y_test, y_pred_test)
    assert np.allclose(cpp, py), "regression_metrics fail"

    cpp = fast_algo.find_peaks(test, LOOKAHEAD, MIN_HEIGHT)
    py = py_find_peaks(test, LOOKAHEAD, MIN_HEIGHT)
    assert set(cpp) == set(py), "find_peaks fail"
    print("✓ Все функции корректны\n")

check_correctness()

# ================== Подробная валидация каждого метода на всех размерах ==================
def validate_all_functions():
    print("=" * 80)
    print("ПОДРОБНАЯ ВАЛИДАЦИЯ НА ВСЕХ РАЗМЕРАХ ДАННЫХ")
    print("=" * 80)

    # moving_average
    print("\n--- moving_average ---")
    for label, data in datasets.items():
        cpp = fast_algo.moving_average(data, WINDOW)
        ref = py_moving_average(data, WINDOW)
        ok = np.allclose(cpp, ref)
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ out[:5]  = {cpp[:5]}")
        print(f"  Ref out[:5]  = {ref[:5]}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # ewma
    print("\n--- ewma ---")
    for label, data in datasets.items():
        cpp = fast_algo.ewma(data, ALPHA)
        ref = py_ewma(data, ALPHA)
        ok = np.allclose(cpp, ref)
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ out[:5]  = {cpp[:5]}")
        print(f"  Ref out[:5]  = {ref[:5]}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # median
    print("\n--- median ---")
    for label, data in datasets.items():
        cpp = fast_algo.median(data)
        ref = py_median(data)
        ok = abs(cpp - ref) < 1e-9
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ median = {cpp}")
        print(f"  Ref median = {ref}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # zscore
    print("\n--- zscore ---")
    for label, data in datasets.items():
        cpp = fast_algo.zscore(data)
        ref = py_zscore(data)
        ok = np.allclose(cpp, ref)
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ out[:5]  = {cpp[:5]}")
        print(f"  Ref out[:5]  = {ref[:5]}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # find_anomalies
    print("\n--- find_anomalies ---")
    for label, data in datasets.items():
        cpp = fast_algo.find_anomalies(data, THRESHOLD)
        ref = py_find_anomalies(data, THRESHOLD)
        ok = cpp == ref
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ anomalies (first 10 indices) = {cpp[:10]}")
        print(f"  Ref anomalies (first 10 indices) = {ref[:10]}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # linear_regression
    print("\n--- linear_regression ---")
    for label, (x, y) in reg_data.items():
        cpp = fast_algo.linear_regression(x, y)
        ref = py_linear_regression(x, y)
        ok = np.allclose(cpp, ref)
        print(f"{label}: x[:5]={x[:5]} y[:5]={y[:5]} ...")
        print(f"  C++ (intercept, slope, r2) = {cpp}")
        print(f"  Ref (intercept, slope, r2) = {ref}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # regression_metrics
    print("\n--- regression_metrics ---")
    for label, (y_true, y_pred) in metrics_data.items():
        cpp = fast_algo.regression_metrics(y_true, y_pred)
        ref = py_regression_metrics(y_true, y_pred)
        ok = np.allclose(cpp, ref)
        print(f"{label}: y_true[:5]={y_true[:5]} y_pred[:5]={y_pred[:5]} ...")
        print(f"  C++ (mae,mse,rmse,r2) = {cpp}")
        print(f"  Ref (mae,mse,rmse,r2) = {ref}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    # find_peaks
    print("\n--- find_peaks ---")
    for label, data in datasets.items():
        cpp = fast_algo.find_peaks(data, LOOKAHEAD, MIN_HEIGHT)
        ref = py_find_peaks(data, LOOKAHEAD, MIN_HEIGHT)
        ok = cpp == ref
        print(f"{label}: input[:5]={data[:5]} ...")
        print(f"  C++ peaks (first 10) = {cpp[:10]}")
        print(f"  Ref peaks (first 10) = {ref[:10]}")
        print(f"  Status: {'✅ OK' if ok else '❌ FAIL'}")

    print("\n" + "=" * 80)
    print("ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)

# Запускаем подробную валидацию перед бенчмарком
validate_all_functions()

# ================== Запуск бенчмарков ==================
results = {}
def run_bench(method_name, lib_name, func, data_iter, adapter=None, args=()):
    times = []
    for label, data in data_iter.items():
        arg = adapter(data) if adapter else data
        t = measure(func, arg, args)
        times.append(t)
    results.setdefault(method_name, {})[lib_name] = times
    print(f"{method_name:20s} {lib_name:25s}: Small={times[0]:.6f}s  Med={times[1]:.6f}s  Large={times[2]:.6f}s")

print("\n" + "=" * 70)
print("БЕНЧМАРК (100k / 500k / 3M)")
print("=" * 70)

# ... (все вызовы run_bench как в предыдущем полном скрипте) ...
# Я повторю их для полноты, но можно вставить свои.

# 1. moving_average
print("\n--- moving_average (window=20) ---")
meth = "moving_average"
run_bench(meth, "C++ (fast_algo)", lambda d: fast_algo.moving_average(d, WINDOW), datasets)
run_bench(meth, "Pure Python", py_moving_average, datasets, args=(WINDOW,))
run_bench(meth, "NumPy convolve", lambda d: np.convolve(d, np.ones(WINDOW)/WINDOW, mode='valid'), datasets, adapter=np.array)
run_bench(meth, "Pandas rolling", lambda d: pd.Series(d).rolling(WINDOW).mean().dropna().tolist(), datasets)

# 2. ewma
print("\n--- ewma (alpha=0.3) ---")
meth = "ewma"
run_bench(meth, "C++ (fast_algo)", lambda d: fast_algo.ewma(d, ALPHA), datasets)
run_bench(meth, "Pure Python", py_ewma, datasets, args=(ALPHA,))
run_bench(meth, "Pandas ewm", lambda d: pd.Series(d).ewm(alpha=ALPHA, adjust=False).mean().tolist(), datasets)

# 3. median
print("\n--- median ---")
meth = "median"
run_bench(meth, "C++ (fast_algo)", fast_algo.median, datasets)
run_bench(meth, "Pure Python", py_median, datasets)
run_bench(meth, "NumPy median", np.median, datasets, adapter=np.array)
run_bench(meth, "Pandas median", lambda d: pd.Series(d).median(), datasets)

# 4. zscore
print("\n--- zscore (ddof=0) ---")
meth = "zscore"
run_bench(meth, "C++ (fast_algo)", fast_algo.zscore, datasets)
run_bench(meth, "Pure Python", py_zscore, datasets)
run_bench(meth, "SciPy zscore", lambda d: stats.zscore(d, ddof=0).tolist(), datasets, adapter=np.array)

# 5. find_anomalies
print("\n--- find_anomalies (threshold=2.5) ---")
meth = "find_anomalies"
run_bench(meth, "C++ (fast_algo)", lambda d: fast_algo.find_anomalies(d, THRESHOLD), datasets)
run_bench(meth, "Pure Python", py_find_anomalies, datasets, args=(THRESHOLD,))
run_bench(meth, "SciPy + list", lambda d: [i for i, z in enumerate(stats.zscore(d, ddof=0)) if abs(z) > THRESHOLD], datasets, adapter=np.array)

# 6. linear_regression
print("\n--- linear_regression ---")
meth = "linear_regression"
for lib_name, func, _, _ in [
    ("C++ (fast_algo)", lambda d: fast_algo.linear_regression(*d), None, ()),
    ("Pure Python", lambda d: py_linear_regression(*d), None, ()),
    ("NumPy polyfit", lambda d: np.polyfit(np.array(d[0]), np.array(d[1]), 1), None, ()),
    ("StatsModels OLS", lambda d: sm.OLS(d[1], sm.add_constant(d[0])).fit().params, None, ())
]:
    times = []
    for label, data in reg_data.items():
        t = measure(func, data)
        times.append(t)
    results.setdefault(meth, {})[lib_name] = times
    print(f"{meth:20s} {lib_name:25s}: Small={times[0]:.6f}s  Med={times[1]:.6f}s  Large={times[2]:.6f}s")

# 7. regression_metrics
print("\n--- regression_metrics ---")
meth = "regression_metrics"
for lib_name, func, _, _ in [
    ("C++ (fast_algo)", lambda d: fast_algo.regression_metrics(*d), None, ()),
    ("Pure Python", lambda d: py_regression_metrics(*d), None, ()),
    ("Scikit-learn", lambda d: (mean_absolute_error(d[0], d[1]), mean_squared_error(d[0], d[1]), np.sqrt(mean_squared_error(d[0], d[1])), sk_r2(d[0], d[1])), None, ())
]:
    times = []
    for label, data in metrics_data.items():
        t = measure(func, data)
        times.append(t)
    results.setdefault(meth, {})[lib_name] = times
    print(f"{meth:20s} {lib_name:25s}: Small={times[0]:.6f}s  Med={times[1]:.6f}s  Large={times[2]:.6f}s")

# 8. find_peaks
print("\n--- find_peaks (lookahead=5, min_height=1.0) ---")
meth = "find_peaks"
run_bench(meth, "C++ (fast_algo)", lambda d: fast_algo.find_peaks(d, LOOKAHEAD, MIN_HEIGHT), datasets)
run_bench(meth, "Pure Python", py_find_peaks, datasets, args=(LOOKAHEAD, MIN_HEIGHT))
run_bench(meth, "SciPy find_peaks", lambda d: signal.find_peaks(d, distance=LOOKAHEAD, prominence=MIN_HEIGHT)[0], datasets, adapter=np.array)

# ================== Markdown таблицы ==================
print("\n\n=== ТАБЛИЦЫ ДЛЯ README (Markdown) ===")
for method, entries in results.items():
    print(f"\n### {method}\n")
    header = "| Library | Small (100k) | Medium (500k) | Large (3M) |"
    sep = "|---------|--------------|---------------|------------|"
    print(header)
    print(sep)
    for lib, times in entries.items():
        print(f"| {lib} | {times[0]:.4f} s | {times[1]:.4f} s | {times[2]:.4f} s |")
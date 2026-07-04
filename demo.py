import fast_algo
import time
data = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0]
for i in range(1,999911):
    data.append(float(i) ** 2 )
    data.append(float(i) )
    data.append(float(i) ** 2 )
    data.append(float(i) )
    data.append(float(i) ** 2 )
    data.append(float(i) )
    data.append(float(i) ** 2 )

def py_find_anomalies(data, threshold=2.5):
    """
    Находит индексы аномалий по z-score порогу.
    
    Параметры:
        data      : list[float | int] – входные числа.
        threshold : float – порог для модуля z-оценки (по умолчанию 3.0).
    
    Возвращает:
        list[int] – индексы элементов, где |z| > threshold.
        Если данных < 2, выбрасывается ValueError (как в C++ zscore).
    """
    # Используем py_zscore (определена выше в твоём скрипте)
    z = py_zscore(data)  # бросает ValueError при len < 2
    anomalies = []
    for i, z_val in enumerate(z):
        if abs(z_val) > threshold:
            anomalies.append(i)
    return anomalies

def py_zscore(data):
    """
    Вычисляет z-оценки для списка чисел.
    Возвращает список float той же длины.
    Если stddev < 1e-10, возвращает список нулей.
    Бросает ValueError, если элементов < 2.
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 values")
    
    n = len(data)
    mean = sum(data) / n
    
    variance_sum = 0.0
    for x in data:
        diff = x - mean
        variance_sum += diff * diff
    stddev = (variance_sum / n) ** 0.5   # смещённая оценка
    
    if stddev < 1e-10:
        return [0.0] * n
    
    return [(x - mean) / stddev for x in data]

def py_median(data):
    """
    Вычисляет медиану, устойчивую к выбросам.
    
    Параметры:
        data : list[float | int] — входные числа.
    
    Возвращает:
        float — медиана.
    
    Выбрасывает ValueError, если список пуст.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Копируем и сортируем (как std::vector<double> sorted = data; std::sort(...))
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if n % 2 == 0:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
    else:
        return sorted_data[n // 2]


def py_ewma(data, alpha):
    """
    Экспоненциально взвешенное скользящее среднее.
    
    Параметры:
        data  : list[float | int] – входные значения.
        alpha : float – коэффициент сглаживания (0 < alpha <= 1).
    
    Возвращает:
        list[float] – сглаженные значения, длина совпадает с data.
        Пустой список, если data пуст.
    
    Формула:
        result[0] = data[0]
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]  для i >= 1
    """
    if not data:
        return []
    
    n = len(data)
    result = [0.0] * n
    result[0] = data[0]
    
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i-1]
    
    return result
6
t1 = time.time()
fast_algo.find_anomalies(data,2.5)
# fast_algo.zscore(data)
# fast_algo.median(data)
# fast_algo.moving_average(data, 3)
# fast_algo.ewma(data, 0.5)
t2 = time.time()
t3 = time.time()

py_find_anomalies(data)
# py_zscore(data)
# py_median(data)
# def py_moving_average(data, window):
#     n = len(data)
#     if window <= 0 or n < window:
#         return []
#     res = [0.0] * (n - window + 1)
#     s = sum(data[:window])
#     res[0] = s / window
#     for i in range(window, n):
#         s += data[i] - data[i - window]
#         res[i - window + 1] = s / window
#     return res
# py_moving_average(data, 3)
# py_ewma(data, 0.5)
t4 = time.time()


x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
intercept, slope, r2 = fast_algo.linear_regression(x, y)
print( intercept, slope, r2)
# print( f"{t2 - t1} алгос плюсов")
# print (f"{t4 - t3} алгос питона")

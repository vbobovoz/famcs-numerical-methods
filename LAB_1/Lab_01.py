import numpy as np


def transpose_matrix(A_):
    size = A_.shape[0]
    transposed_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            transposed_matrix[j, i] = A_[i, j]
    return transposed_matrix

def is_symmetric(A_):
    size = A_.shape[0]
    for i in range(size):
        for j in range(size):
            if A_[i][j] != A_[j][i]:
                return False
    return True

def gauss_transformation(A_, b_):
    if is_symmetric(A_):
        return A_, b_
    else:
        return np.dot(transpose_matrix(A_), A_), np.dot(transpose_matrix(A_), b_)

def gauss_seidel(A_, b_, max_iterations=1000, tolerance=1e-6):
    size = A_.shape[0]
    A_, b_ = gauss_transformation(A_, b_)

    x_ = np.zeros(size)
    for k in range(max_iterations):
        x_new = np.copy(x_)
        for i in range(size):
            sigma = np.dot(A_[i, :i], x_new[:i]) + np.dot(A_[i, i + 1:], x_[i + 1:])
            x_new[i] = (1 / A_[i, i]) * (b_[i] - sigma)

        if np.linalg.norm(x_new - x_) < tolerance:
            return x_new

        x_ = x_new

    raise ValueError("Метод релаксации не сошелся за максимальное число итераций")

def f_x0(polynom, x0):
    value = 0
    for i in range(len(polynom)):
        value += polynom[i] * (x0**i)
    return value

def diff_f_x0(polynom, x0):
    new_poly = []
    for i in range(1, len(polynom)):
        el_poly = polynom[i] * i * (x0**(i-1))
        new_poly.append(el_poly)
    return sum(new_poly)

def newton(polynom, x0, eps=1e-6):
    iter_counter = 0
    while True:
        xk = x0 - f_x0(polynom, x0) / diff_f_x0(polynom, x0)
        iter_counter += 1
        if abs(x0 - xk) >= eps:
            x0 = xk
        else:
            break
    return xk

def horner_scheme(poly, x0):
    n = len(poly)
    result = [0] * (n - 1)  # Инициализируем новый полином нулями

    # Проходим по коэффициентам, начиная с последнего
    for i in range(n - 1, 0, -1):
        result[i - 1] = poly[i] + (result[i] if i < n - 1 else 0) * x0

    return result

def roots_of_poly(polynom, x0=0):
    roots = []
    while True:
        if len(polynom) == 3:
            # Если у нас осталось квадратное уравнение
            a, b, c = polynom
            D = b ** 2 - 4 * a * c
            if D < 0:
                # Корней нет
                break
            else:
                root1 = (-b + D ** 0.5) / (2 * a)
                root2 = (-b - D ** 0.5) / (2 * a)
                print("D, ", root1, root2)
                roots.append(root1)
                roots.append(root2)
                break
        else:
            root = newton(polynom, x0)
            roots.append(root)
            polynom = horner_scheme(polynom, root)
    return roots


if __name__ == '__main__':
    # Инициализация матрицы A
    A = np.array([[5, -2, 0.5, 1, -0.1],
                  [1, -8, -3.1, 1, 2.3],
                  [-1, 3, 10, 2, 4.6],
                  [0, 0.1, 2, 15, 2],
                  [1, -2, 0.4, 3.2, -17]])

    # Инициализация вектора f
    b = np.array([1, 0, 2, -3, 5])

    # Получение коэффициентов путем решения СЛАУ
    a = gauss_seidel(A, b).tolist()

    # Вывод полинома:
    poly = ""
    for i in range(5):
        poly += f"({a[4-i]} * x^{4-i}) + "
    poly = poly[:-3]
    print(f"После решения СЛАУ, получим коэффициенты, из которых получим многочлен следующего вида:\n{poly}")

    # Решение многочлена:
    # -0.2870922787947789 -0.22238701134031116 0.4691074958043837 -0.282250798823043 0.07892474533288997
    # 0.564055 -1.926292
    roots = roots_of_poly(a)
    print(f"Корни полученного многочлена: {roots}")
    print(f"Максимальный по модулю корень: {np.max(np.abs(roots))}")
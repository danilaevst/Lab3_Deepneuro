import numpy as np
import random
import torch

#  Номер по списку группы (замените на свой)
your_list_number = 6  # Четный номер (не нужно проверять)

# 1. Создание целочисленного тензора, преобразование в float32 и включение отслеживания градиента
x = torch.randint(0, 10, (5, 3), dtype=torch.int32).float()
x.requires_grad_(True)
print("x:\n", x)

# 2. Возведение в степень n
n = 3  # Номер 6 - четный, значит n=3
y = x ** n
print(f"x^{n}:\n", y)

# 3. Умножение на случайное значение от 1 до 10
a = random.randint(1, 10)
z = y * a
print("Случайное число a =", a)
print("z = x^n * a:\n", z)

# 4. Взятие экспоненты
result = torch.exp(z)
torch.set_printoptions(precision=4, sci_mode=False)
print("exp(z):\n", result)

# 5. Вычисление производной по x
result.sum().backward()
print("Градиент по x (dy/dx):\n", x.grad)

# WineQualityMLOpsMVP/tests/test_example_utils.py

# Una función de ejemplo que podríamos querer probar
def format_greeting(name):
    if not name:
        return "Hello, Guest!"
    return f"Hello, {name}!"

# Otra función de ejemplo
def calculate_sum(numbers):
    return sum(numbers)

# Pruebas unitarias usando pytest (deben empezar con test_)
def test_format_greeting_with_name():
    assert format_greeting("Alice") == "Hello, Alice!"

def test_format_greeting_no_name():
    assert format_greeting("") == "Hello, Guest!"
    assert format_greeting(None) == "Hello, Guest!"

def test_calculate_sum_positive_numbers():
    assert calculate_sum([1, 2, 3, 4]) == 10

def test_calculate_sum_empty_list():
    assert calculate_sum([]) == 0

def test_calculate_sum_with_negatives():
    assert calculate_sum([-1, -2, 3, 0]) == 0
    
##Esta es una prueba
##Esta es otra prueba
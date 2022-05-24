# Запуск проекта из PyCharm

1. Установить необходимое ПО
    1. Установить python 3.7: https://www.python.org/downloads/release/python-370/
    1. Запустить терминал от имени администратора
    1. Перейти в папку проекта
    1. Создать виртуальное окружение для запуска выполнив `python -m venv venv`
    1. Активировать созданное виртуальное окружение выполнив: `venv\Scripts\activate.bat`
    1. Убедиться, что находишься в виртуальном окружении: в терминале строка должна начинаться с (venv)
    1. Установить библиотеки выполнив `pip install -r requirements.txt`

# Запуск проекта

1. Выполните `python main.py --rn 4 --ps 6 --sys 2`, где:
    1. rn - reconciliation number, количество последних значений для сверки
    1. ps - predict steps, количество значений для прогнозирования
    1. sys - using system, используемая система:
        1. 1 - single lstm
        1. 2 - multi lstm

# Ссылки

- типы нейросетей: https://python-school.ru/blog/types-of-neural-nets/

- реализация рекурентной: https://python-scripts.com/recurrent-neural-network

- описание базовых принципов: https://python-scripts.com/intro-to-neural-networks

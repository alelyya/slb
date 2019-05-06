[Render in nbviewer...](https://nbviewer.jupyter.org/github/alelyya/slb/blob/master/v1/v1.ipynb?flush_cache=true)

usage: segment.py [-h] [-o OUTPUT_FOLDER] [-v] [filename [filename ...]]

positional arguments:
  filename

optional arguments:
  -h, --help        show this help message and exit
  -o OUTPUT_FOLDER  ./results/ if not specified
  -v                save all intermediate results

На вход подаётся изображение в оттенках серого – светлые частицы на тёмном фоне (см. пример ниже; сами оттенки могут меняться, но частицы всегда светлее фона). Необходимо отделить частицы от фона и оценить их размер (в пикселях).

<img src="https://raw.githubusercontent.com/alelyya/slb/master/v1/0.png" width="500" height="500">

В результате работы алгоритма для любого корректного изображения в оттенках серого автоматизированным образом должны быть получены (т.е. без участия оператора):
1) Чёрно-белое изображение «Фон и частицы».
2) Разметить каждую частицу как отдельный объект.
3) Предложить метрику и оценить размер каждой частицы. Размер должен считаться единым способом для всех частиц.
4) Построить распределение частиц по размерам.
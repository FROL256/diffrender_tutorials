# Проект по созданию кросс-платформенного дифференцируемого рендера

## Быстрая сборка

Сборка c нуля Enzyme на Ubuntu23 и clang-15 в папке **'external/enzyme/enzyme/build'**:
* sudo apt install llvm
* sudo apt install clang
* sudo apt install llvm-dev
* sudo apt install libclang-dev
* sudo apt install zlib1g zlib1g-dev -y
* cmake -G Ninja .. -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm -DClang_DIR=/usr/lib/cmake/clang-15
* ninja

Сборка с нуля самого рендера в папке **'cmake-build-release'** (например):
* sudo apt install libomp-dev
* cmake .. -DCMAKE_BUILD_TYPE=Release -DDEBUG=OFF -DUSE_EMBREE=ON -DUSE_OPENMP=ON -DUSE_MITSUBA=OFF -DUSE_ENZYME=ON -DCLANG_VERSION=15
* make -j 8

## Более подробно о зависимостях

Все зависимости проекта на данный момент опциональны. Вы можете собрать программу
без них.

1) Базовый вариант (без зависимостей)
 - установить clang
 - cmake CMakeLists.txt -DDEBUG=OFF -DUSE_EMBREE=OFF -USE_OPENMP=OFF -DUSE_MITSUBA=OFF -DUSE_ENZYME=OFF -DCLANG_VERSION=9
 - версия clang может быть другой, но не ниже 9
2) Embree - самая критичная зависимость, без нее скорость работы рендера ОЧЕНЬ низкая
 - по идее, никаких дополнительных действий не требуется, только установить -DUSE_EMBREE=ON
3) OpenMP - существенно ускоряет вычисления на CPU
 - для clang в данный момент нужно указывать в CMake две переменные: AUX_OMP_INC и AUX_OMP_LIB
 - нужные пути можно получить при помощи команд 'locate omp.h'' и 'locate libomp.so' 
4) Mitsuba - сейчас используется только в тестах, как референс для нашего рендера
 - установить python (проверено на версии 3.8)
 - python -m pip install mitsuba
5) Enzyme - библиотека автоматического дифференцирования
 - установить llvm (должно поставиться вместе с clang)
 - собрать enzyme https://enzyme.mit.edu/Installation/
   * исходники уже лежат в external/enzyme
   * на Ubuntu 23 можно использовать такую команду: 'cmake -G Ninja .. -DLLVM_DIR=/usr/lib/llvm-16/share/llvm/cmake/ -DClang_DIR=/usr/lib/llvm-16/lib/cmake/clang/'  
 - указать правильную версию clang -DCLANG_VERSION=...

Рекомендуемая сборка:
cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Release -DDEBUG=OFF -DUSE_EMBREE=ON -DUSE_OPENMP=ON -DUSE_MITSUBA=ON -DUSE_ENZYME=ON -DCLANG_VERSION=9


## Картиночки

Примеры тестоовых оптимизаций:<br>
<img src="gifs/cube.gif"/> <br>
<img src="gifs/pyramid.gif"/> <br>
<img src="gifs/sphere.gif"/> <br>



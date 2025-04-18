CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -O3 -DNDEBUG -march=native -mtune=native 
INCLUDES = -Iinclude -Ilibnpy/include -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/eigen3

build/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin/main: build/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@

default: bin/main

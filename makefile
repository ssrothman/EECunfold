CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -O3 -march=native -mtune=native -fopenmp -fPIC -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACK -lblas -llapack -ffast-math -flto=8
INCLUDES = -Iinclude -Ilibnpy/include -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/eigen3

HEADERS = $(wildcard include/*.h)
SOURCES = $(wildcard src/*.cc)
OBJECTS = $(patsubst src/%.cc, build/%.o, $(SOURCES))

default: bin/main

build/exists:
	mkdir -p build
	touch $@

bin/exists:
	mkdir -p bin
	touch $@

build/%.o: src/%.cc $(HEADERS) build/exists
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

bin/main: $(OBJECTS) bin/exists
	$(CXX) $(CXXFLAGS) $^ -o $@


clean:
	rm -f build/* bin/*
	touch build/exists bin/exists

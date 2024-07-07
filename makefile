CPPFLAGS = -std=c++17 
CC = g++
INCLUDES = -I./simon_util_cpp -I./libnpy/include -I/home/simon/miniforge3/include/eigen3 -I/work/submit/srothman/miniforge3/envs/updated/include/eigen3
LIBRARIES = -llapacke -llapack -lblas -lstdc++
LD_FLAGS = -L/home/simon/miniforge3/lib -L/work/submit/srothman/miniforge3/envs/updated/lib/

OPTIM_FLAGS = -O3 -march=native -mtune=native -flto=auto

default: main main_optim

main.o : main.cc simon_util_cpp/*.h
	$(CC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

main_optim.o: main.cc simon_util_cpp/*.h
	$(CC) $(CPPFLAGS) $(OPTIM_FLAGS) $(INCLUDES) -c $< -o $@

main: main.o
	g++ main.o -o main $(LD_FLAGS) $(LIBRARIES) $(CPPFLAGS)

main_optim: main_optim.o
	g++ main_optim.o -o main_optim $(LD_FLAGS) $(LIBRARIES) $(CPPFLAGS) $(OPTIM_FLAGS)
	
clean:
	rm -f main main_optim main.o main_optim.o

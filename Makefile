all:
	mpicxx -o pqsort test.cpp
	mpirun -np 6 ./pqsort input output
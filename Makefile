OBJ = 	src/Vector.o src/Matrix.o
CPP_INCLUDE = -Iinclude
CPP_FLAGS = -g

%.o: %.cpp
	g++ -c $< ${CPP_INCLUDE} ${CPP_FLAGS} -o $@

default: $(OBJ)
	g++ -c test/LinAlg.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/LinAlg.o
	g++ test/LinAlg.o ${OBJ} -o linalg_test

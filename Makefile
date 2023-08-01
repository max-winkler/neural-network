OBJ = 	src/Vector.o src/Matrix.o src/NeuralNetwork.o src/Activation.o src/TrainingData.o
CPP_INCLUDE = -Iinclude
CPP_FLAGS = -O3
LIBS = -lpng

%.o: %.cpp
	g++ -c $< ${CPP_INCLUDE} ${CPP_FLAGS} -o $@

default: $(OBJ)
	g++ -c test/LinAlg.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/LinAlg.o
	g++ test/LinAlg.o ${OBJ} -o linalg_test
	g++ -c test/NeuralNetwork.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/NeuralNetwork.o	
	g++ test/NeuralNetwork.o ${OBJ} -o neuralnetwork_test
	g++ -c test/Classification.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/Classification.o	
	g++ test/Classification.o ${OBJ} ${LIBS} -o classification_test

clean:
	rm src/*.o

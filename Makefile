OBJ = 	src/DataArray.o src/Vector.o src/Matrix.o src/Layer.o src/NeuralNetwork.o src/Activation.o src/TrainingData.o
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
	g++ -c test/Convolution.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/Convolution.o	
	g++ test/Convolution.o ${OBJ} ${LIBS} -o convolution_test
	g++ -c test/DigitRecognition.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/DigitRecognition.o	
	g++ test/DigitRecognition.o ${OBJ} ${LIBS} -o digitrecognition_test

clean:
	rm src/*.o

OBJ = 	src/DataArray.o \
	src/Vector.o \
	src/Matrix.o \
	src/Random.o \
	src/Layer.o \
	src/VectorInputLayer.o \
	src/MatrixInputLayer.o \
	src/FullyConnectedLayer.o \
	src/NeuralNetwork.o \
	src/Activation.o \
	src/TrainingData.o

TESTS = test/DigitRecognition.o
CPP_INCLUDE = -Iinclude
CPP_FLAGS = -g
LIBS = -lpng

MNIST_FILES_URL := http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
	http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

MNIST_FILE_DIR := mnist

MNIST_FILES_GZ := $(addprefix $(MNIST_FILE_DIR)/,$(notdir $(MNIST_FILES_URL)))
MNIST_FILES := $(addprefix $(MNIST_FILE_DIR)/,$(patsubst %.gz,%,$(notdir $(MNIST_FILES_URL))))
EXISTING_FILES := $(wildcard $(MNIST_FILES))

%.o: %.cpp
	g++ -c $< ${CPP_INCLUDE} ${CPP_FLAGS} -o $@

test: $(OBJ) test/NeuralNetwork.cpp
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
	g++ -c test/Pooling.cpp ${CPP_INCLUDE} ${CPP_FLAGS} -o test/Pooling.o	
	g++ test/Pooling.o ${OBJ} ${LIBS} -o pooling_test

# Target to create the output directory
$(MNIST_FILE_DIR):
	mkdir -p $(MNIST_FILE_DIR)

# Target to download all files
$(MNIST_FILES_GZ): $(MNIST_FILE_DIR) | $(EXISTING_FILES)
	@for url in $(MNIST_FILES_URL); do \
		file=$$(basename $$url); \
		wget -P $(MNIST_FILE_DIR) -O $(MNIST_FILE_DIR)/$$file $$url; \
	done

# Target rule to extract gzip files
$(MNIST_FILES): $(MNIST_FILES_GZ)
	@for file in $(MNIST_FILES_GZ); do \
		gunzip -k $$file; \
	done

ressources: $(MNIST_FILES)

default: $(OBJ)

clean:
	rm -f src/*.o mnist/*.gz

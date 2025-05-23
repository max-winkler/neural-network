OBJ = 	src/DataArray.o \
	src/Vector.o \
	src/Matrix.o \
	src/Tensor.o \
	src/LinAlg.o \
	src/Random.o \
	src/Layer.o \
	src/VectorInputLayer.o \
	src/MatrixInputLayer.o \
	src/FullyConnectedLayer.o \
	src/FlatteningLayer.o \
	src/ConvolutionalLayer.o \
	src/PoolingLayer.o \
	src/LayerFactory.o \
	src/NeuralNetwork.o \
	src/Activation.o \
	src/TrainingData.o \
	src/Image.o

TESTS = 	test/LinAlg.o \
	test/DigitRecognition.o \
	test/Classification.o \
	test/NeuralNetwork.o \
	test/Convolution.o \
	test/Pooling.o

CPP_INCLUDE = -Iinclude
CPP_FLAGS = -g -O0
# CPP_FLAGS = -O3
LIBS = -lpng -lblas -lpugixml

CBLAS_CHECK := $(shell echo 'int main(){}' | $(CXX) -x c++ - -lcblas -lblas -o /dev/null 2>/dev/null && echo yes || echo no)
ifeq ($(CBLAS_CHECK),yes)
  LIBS += -lcblas
endif


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


test: $(OBJ) $(TESTS)
	g++ test/LinAlg.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o linalg_test
	g++ test/NeuralNetwork.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o neuralnetwork_test
	g++ test/Classification.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o classification_test
	g++ test/Convolution.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o convolution_test
	g++ test/DigitRecognition.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o digitrecognition_test
	g++ test/Pooling.o ${OBJ} ${LIBS} ${CPP_FLAGS} -o pooling_test

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

default: $(OBJ) test

clean:
	rm -f src/*.o test/*.o mnist/*.gz

ROCM_PATH = /opt/rocm
HIPCC = $(ROCM_PATH)/bin/hipcc
ROCM_LIB = $(ROCM_PATH)/lib

CXXFLAGS = -std=c++20 -O2 -g -Wall -Wextra
HIP_FLAGS = --offload-arch=gfx1201
INCLUDE_FLAGS = -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/rocwmma -I.
LIBRARY_FLAGS = -L$(ROCM_LIB)

BUILD_DIR = ./build
TARGET = $(BUILD_DIR)/bin
KERNEL_SRC = kernels.cpp
KERNEL_HDR = kernels.hpp
UTILS_HDR = utils.hpp

SOURCES = main.cpp $(KERNEL_SRC)
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS)
	$(HIPCC) $(CXXFLAGS) $(HIP_FLAGS) -o $@ $^ $(LIBRARY_FLAGS)

$(BUILD_DIR)/%.o: %.cpp $(KERNEL_HDR) $(UTILS_HDR) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $(HIP_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

$(BUILD_DIR)/main.o: $(KERNEL_HDR) $(UTILS_HDR)
$(BUILD_DIR)/kernels.o: $(KERNEL_HDR) $(UTILS_HDR)

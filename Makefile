TARGET					:= bin
ROCM_INSTALL_DIR		:= /opt/rocm
HIPCC					:= $(ROCM_INSTALL_DIR)/bin/hipcc
CXX_FLAGS 				:= -Wall -Wextra -O3 -std=c++20
INCLUDE_DIR             := -I $(ROCM_INSTALL_DIR)/include -I $(ROCM_INSTALL_DIR)/include/rocwmma
LDLIB					:= -L $(ROCM_INSTALL_DIR)/lib
GFX						:= gfx1201

SOURCE                  := main.cpp

all: $(TARGET)

.PHONY: all run clean

$(TARGET): $(SOURCE)
	$(HIPCC) $(INCLUDE_DIR) $(LDLIB) $(CXX_FLAGS) --offload-arch=$(GFX) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET)


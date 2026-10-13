CC = gcc
CFLAGS = -std=c17 -Wall -Wextra -O2

ifeq ($(OS),Windows_NT)
	AF_DEFAULT_DIR = C:/Program Files/ArrayFire/v3
else
	AF_DEFAULT_DIR = /opt/arrayfire
	LDLIBS_EXTRA = -lm
	AF_RUNTIME_LIB_DIR = $(if $(filter /%,$(firstword $(AF_LIB_DIR))),$(AF_LIB_DIR),$(CURDIR)/$(AF_LIB_DIR))
	LDFLAGS_EXTRA = -Wl,--disable-new-dtags,-rpath,"$(AF_RUNTIME_LIB_DIR)"
endif

# AF_DIR: argomento di make o variabile ambiente; altrimenti AF_PATH o percorso standard.
AF_DIR ?= $(if $(strip $(AF_PATH)),$(subst \,/,$(AF_PATH)),$(AF_DEFAULT_DIR))
AF_LIB_DIR ?= $(if $(wildcard $(AF_DIR)/lib64/libafcpu.so*),$(AF_DIR)/lib64,$(AF_DIR)/lib)


CPPFLAGS = -I"$(AF_DIR)/include"     			#serve per trovare i file include
LDFLAGS = -L"$(AF_LIB_DIR)" $(LDFLAGS_EXTRA)	#percorsi delle librerie per linking e runtime
LDLIBS = -lafcpu $(LDLIBS_EXTRA)				#serve per far trovare la singola libreria(in c ci sono i .lib)

TARGET = frame_work
OBJ = main.o graph.o activation_function.o loss.o backpropagation.o MLP.o optimizer.o

#$@  nome dell’obiettivo da generare.
#$<  nome della prima dipendenza.

.PHONY: all build rebuild cleanAll clean setup-windows setup-linux

build: $(TARGET)

setup-windows:
	cmd /c "scripts\download_arrayfire.bat"

setup-linux:
	bash scripts/download_arrayfire.sh

rebuild:
	$(MAKE) cleanAll
	$(MAKE) build

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

main.o: main.c autograd/graph.h loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

graph.o: autograd/graph.c autograd/graph.h loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

activation_function.o: activation_functions/activation_function.c activation_functions/activation_function.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

loss.o: loss/loss.c loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

backpropagation.o: autograd/backpropagation.c autograd/backpropagation.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

MLP.o: models/MLP/MLP.c models/MLP/MLP.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

optimizer.o: optimizers/optimizer.c optimizers/optimizer.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

cleanAll:
ifeq ($(OS),Windows_NT)
	powershell.exe -NoProfile -Command "Get-ChildItem -Path './*' -Include '*.o', '$(TARGET)' -File | Remove-Item -Force -ErrorAction Stop"
else
	$(RM) $(OBJ) $(TARGET)
endif

clean:
ifeq ($(OS),Windows_NT)
	powershell.exe -NoProfile -Command "Get-ChildItem -Path './*' -Include '*.o' -File | Remove-Item -Force -ErrorAction Stop"
else
	$(RM) $(OBJ)
endif


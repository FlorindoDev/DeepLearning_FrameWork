CC = gcc
CFLAGS = -std=c17 -Wall -Wextra -O2
AF_DIR = C:/Program Files/ArrayFire/v3
CPPFLAGS = -I"$(AF_DIR)/include"     		#serve per trovare i file include
LDFLAGS = -L"$(AF_DIR)/lib"					#serve per far trovare i file c al linker
LDLIBS = -lafcpu							#serve per far trovare la singola libreria(in c ci sono i .lib)

TARGET = frame_work.exe
OBJ = main.o graph.o activation_function.o loss.o backpropagation.o

#$@  nome dell’obiettivo da generare.
#$<  nome della prima dipendenza.

.PHONY: all clean			#Dice a Make che all e clean sono nomi di azioni, non file da generare.

build: $(TARGET)

rebuild:
	$(MAKE) cleanAll
	$(MAKE) build

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

main.o: main.c autograd/graph.h loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

graph.o: autograd/graph.c autograd/graph.h loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

activation_function.o: activation_functions\activation_function.c activation_functions\activation_function.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

loss.o: loss/loss.c loss/loss.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

backpropagation.o: autograd/backpropagation.c autograd/backpropagation.h general.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

cleanAll:
	powershell.exe -NoProfile -Command "Get-ChildItem -Path './*' -Include '*.o', '$(TARGET)' -File | Remove-Item -Force -ErrorAction Stop"

clean:
	powershell.exe -NoProfile -Command "Get-ChildItem -Path './*' -Include '*.o' -File | Remove-Item -Force -ErrorAction Stop"


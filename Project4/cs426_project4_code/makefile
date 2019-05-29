all: gpummultexec.o 
# cpummultexec.o utils.o 
gpummultexec.o:
	nvcc utils.c mmmult.cu -o gpummultexec.o
#cpummultexec.o:
#	cc main.c -o cpummultexec.o
#utils.o:
#	cc utils.h utils.c -o utils.o
clean:
	rm *.o mmultexec

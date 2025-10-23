CC = gcc
LIBDIR = -L/usr/X11R6/lib
CFLAGS = -ansi -pedantic -Wall

all:	xmaxtree

clean:
	rm *~ xmaxtree *.o

xmaxtree: main.o gui.o tlgui.o maxtree.o attrs.o mappers.o ran3.o
	$(CC) -o xmaxtree main.o gui.o tlgui.o maxtree.o attrs.o mappers.o ran3.o $(LIBDIR) -lm -lX11
	strip xmaxtree

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

gui.o: gui.c
	$(CC) $(CFLAGS) -c gui.c

tlgui.o: tlgui.c
	$(CC) $(CFLAGS) -c tlgui.c

maxtree.o: maxtree.c
	$(CC) $(CFLAGS) -c maxtree.c

attrs.o: attrs.c
	$(CC) $(CFLAGS) -c attrs.c

mappers.o: mappers.c
	$(CC) $(CFLAGS) -c mappers.c

ran3.o: ran3.c
	$(CC) $(CFLAGS) -c ran3.c

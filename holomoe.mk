# HoloMoE build.   make -f holomoe.mk
CC ?= cc
UNAME := $(shell uname)
ARCH  := $(shell uname -m)
ifeq ($(UNAME),Darwin)
  PAR :=
else
  PAR := -fopenmp
endif
# Intel/AMD: tune for this CPU (enables AVX2 where available). ARM (Apple Silicon, Pi): portable flags.
ifneq (,$(filter x86_64 i686 i386,$(ARCH)))
  OPT := -O3 -march=native
else
  OPT := -O3
endif

all: holopress holomoe

holopress: holopress.c
	$(CC) -O2 -std=c11 -Wall -o $@ $< -lm

holomoe: holomoe.c
	$(CC) $(OPT) -std=c11 -Wall $(PAR) -o $@ $< -lm

.PHONY: all

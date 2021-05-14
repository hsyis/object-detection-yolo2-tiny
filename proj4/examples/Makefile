SHELL=/bin/bash
SUB_DIR = $(shell ls -d -- */)

all:
	@for d in $(SUB_DIR); do pushd $$d; make; popd; done

clean:
	@for d in $(SUB_DIR); do pushd $$d; make $@; popd; done

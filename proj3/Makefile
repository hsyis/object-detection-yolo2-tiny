.PHONY: run_avx run_avx_debug
run_avx: lib_dnn_avx.so
	cp dnn_avx.py dnn.py
	python3 __init__.py sample.jpg output.jpg

run_avx_debug: lib_dnn_avx.so
	cp dnn_avx.py dnn.py
	python3 __init__.py sample.jpg output.jpg --debug

lib_dnn_avx.so: dnn_avx.c
	gcc -Ofast -march='broadwell' -shared -fPIC -o $@ $^ -mavx2 -mfma -lpthread

.PHONY: run_openblas run_openblas_debug
run_openblas: lib_dnn_openblas.so
	cp dnn_openblas.py dnn.py
	python3 __init__.py sample.jpg output.jpg

run_openblas_debug: lib_dnn_openblas.so
	cp dnn_openblas.py dnn.py
	python3 __init__.py sample.jpg output.jpg --debug

lib_dnn_openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lopenblas

.PHONY: clean
clean:
	rm -f lib_dnn_avx.so lib_dnn_openblas.so



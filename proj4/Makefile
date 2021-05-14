.PHONY: run_cuda run_cuda_debug
run_cuda: lib_dnn_cuda.so
	cp dnn_cuda.py dnn.py
	python3 __init__.py sample.jpg output.jpg

run_cuda_debug: lib_dnn_cuda.so
	cp dnn_cuda.py dnn.py
	python3 __init__.py sample.jpg output.jpg --debug

lib_dnn_cuda.so: dnn_cuda.cu
	nvcc -Xcompiler -fPIC -shared -O3 -o $@ $^

.PHONY: run_cublas run_cublas_debug
run_cublas: lib_dnn_cublas.so
	cp dnn_cublas.py dnn.py
	python3 __init__.py sample.jpg output.jpg

run_cublas_debug: lib_dnn_cublas.so
	cp dnn_cublas.py dnn.py
	python3 __init__.py sample.jpg output.jpg --debug

lib_dnn_cublas.so: dnn_cublas.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

.PHONY: clean
clean:
	rm -f lib_dnn_cuda.so lib_dnn_cublas.so



.PHONY: run_fp32 run_fp16 run_int8
run_fp32: lib_dnn_fp32.so
	python3 __init__.py ../sample.jpg output.jpg FP32

run_fp16: lib_dnn_fp16.so
	python3 __init__.py ../sample.jpg output.jpg FP16

run_int8: lib_dnn_int8.so
	python3 __init__.py ../sample.jpg output.jpg INT8

lib_dnn_fp32.so: dnn_fp32.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

lib_dnn_fp16.so: dnn_fp16.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

lib_dnn_int8.so: dnn_int8.cu
	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

.PHONY: clean
clean:
	rm -f *.so



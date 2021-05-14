all: test_cublas test_cublas_fp16

test_cublas: cublas_example.cu
	nvcc -o $@ $^ -lcublas

test_cublas_fp16: cublas_example_fp16.cu
	nvcc -o $@ $^ -lcublas

clean:
	rm -f test_cublas test_cublas_fp16

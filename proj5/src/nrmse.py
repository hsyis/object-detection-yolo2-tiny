import numpy as np

fp32 = np.load('FP32.npy')
fp16 = np.load('FP16.npy')
int8 = np.load('INT8.npy')
rand = np.random.rand(13, 13, 125)

fp32_max = np.max(fp32)
fp16_max = np.max(fp16)
int8_max = np.max(int8)
rand_max = np.max(rand)

fp32_min = np.min(fp32)
fp16_min = np.min(fp16)
int8_min = np.min(int8)
rand_min = np.min(rand)

fp32_norm = np.linalg.norm(fp32 / fp32_max) * fp32_max
fp16_norm = np.linalg.norm(fp16 / fp16_max) * fp16_max
int8_norm = np.linalg.norm(int8 / int8_max) * int8_max
rand_norm = np.linalg.norm(rand / rand_max) * rand_max

fp32_nrmse = np.sqrt(np.sum(np.square(fp32 - fp32)) / fp32_norm) / (fp32_max - fp32_min)
fp16_nrmse = np.sqrt(np.sum(np.square(fp16 - fp32)) / fp16_norm) / (fp32_max - fp32_min)
int8_nrmse = np.sqrt(np.sum(np.square(int8 - fp32)) / int8_norm) / (fp32_max - fp32_min)
rand_nrmse = np.sqrt(np.sum(np.square(rand - fp32)) / rand_norm) / (fp32_max - fp32_min)

print("fp32: ", fp32_nrmse)
print("fp16: ", fp16_nrmse)
print("int8: ", int8_nrmse)
print("rand: ", rand_nrmse)

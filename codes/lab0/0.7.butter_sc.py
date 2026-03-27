from scipy.signal import butter
b, a = butter(4, 0.2)
print(a,b)


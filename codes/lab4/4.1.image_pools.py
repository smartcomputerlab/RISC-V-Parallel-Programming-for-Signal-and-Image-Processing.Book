from multiprocessing import Pool
import numpy as np

def process_image(i):
    img = np.random.rand(512,512)
    return np.sum(img)
with Pool(4) as p:
    results = p.map(process_image, range(10))
    print(results)


import tensorflow as tf
import Image
import numpy as np
def pre(path):
	threshold = 195
	im = Image.open(path)
	im1 = im.resize((28,28),1)
	im2 = im1.convert("L")
	arr = np.array(im2)
	for i in range(28):
		for j in range(28):
			if arr[i][j] > threshold:
				arr[i][j]=0
			else: arr[i][j]=255
	im3 = Image.fromarray(np.uint8(arr))
	im4 = Image.fromarray(reverse_pool(np.uint8(arr)))
	im4.show()
	xx = arr.reshape([1,784])
	xx1 = xx.astype(np.float32)
	xx2 = np.multiply(xx1, 1.0/255.0)
	#print(xx2)
	return xx2

def reverse_pool(na28):
        na224 = np.zeros((224,224))
        a,b = np.shape(na28)
        for i in range(a):
                for j in range(b):
                        for m in range(8):
                                for n in range(8):
                                        na224[i*8+m][j*8+n] = na28[i][j]
        return na224

def main(argv=None):
        path = input("input path:")
        pre(path)

if __name__=="__main__":
        main()


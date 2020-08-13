import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from math import*
from scipy.fftpack import fft

#load file
array= np.loadtxt('/home/sigma314159265/SWAN/ch00_B0833-45_20150612_191438_010_4')
north=array[:,0]
south=array[:,1]
'''	
plt.plot(north)
plt.title('North Array')
plt.xlabel('')
plt.ylabel('')
plt.show()
'''
array2d=[]

for i in range(1,257):
	ffty=fft(north[(i-1)*512: i*512-1])	
	half=ffty[256:511]
	realpart1= np.square(half.real)
	imagpart1=np.square(half.imag)
	amp1=np.sqrt(realpart1+ imagpart1)
	array2d.append(amp1)

array2dx=np.vstack(array2d)
#print(sizeof(array2d))
#array2d=np.reshape(a,(10,256))


#Plot FFT of North
plt.imshow(array2dx)
plt.xlabel('Frequency Channels 512/33 us each')
plt.ylabel('Amp')
plt.title('FFT of North Array data')
#plt.savefig('vela_north.png')
plt.show()


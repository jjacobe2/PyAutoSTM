import numpy as np
import pySPM
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma

def stm_quality_check(image, threshold=0.002):
    ''' Function to check if image is good scan vs bad scan using z channel data. Checks change
    in z with respect to x and y
    '''
    
    sigma = estimate_sigma(image)
    print(sigma)
    if sigma > threshold:
        print(f'Low quality/Noisy Image')
        
        return False

    return True
    
def auto_correlate(image):
    ''' Perform an auto correlation function using Fast Fourier Transform and Inverse Fast Fourier Transform
    '''
    
    image_ft = np.fft.fft(image, axis=0)
    image_ac = np.fft.ifft(image_ft * np.conjugate(image_ft), axis=0).real
    
    image_ac = np.fft.fft(image_ac, axis=1)
    image_ac = np.fft.ifft(image_ac * np.conjugate(image_ac), axis=1).real
    
    print(np.amax(image_ac))

    return image_ac
    
if __name__ == "__main__":
    img_name = "../../sample2/Topo014.sxm"
    
    image = pySPM.SXM(img_name)
    image_z = image.get_channel(name='Z').pixels
    image_z = pySPM.normalize(image_z)
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_z)
    plt.show()
    
    cont = stm_quality_check(image_z)
    
    if cont:
        print(f'Good Image! Proceeding with AutoAssembly')
    
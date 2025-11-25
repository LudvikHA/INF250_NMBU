import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def histogram(brightnessMatrix: np.ndarray) -> None:
    """
    Display a histogram of pixel brightness values for a grayscale image.

    Parameters
    ----------
    brightnessMatrix : np.ndarray
        A 2D NumPy array representing the grayscale image, where each element 
        is an integer brightness value in the range [0, 255].

    Returns
    -------
    None
        This function does not return a value. It generates and displays a 
        histogram plot showing the distribution of brightness values across 
        the image.
    """
    plt.hist(
        brightnessMatrix.flatten(),
        bins=256,
        color="mediumslateblue",
        linewidth=1.2
    )

    plt.title("Histogram of Brightness Values", fontsize=14, pad=15)
    plt.xlabel("Brightness", fontsize=12)
    plt.ylabel("Pixel Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def otsu(brightnessMatrix: np.ndarray) -> np.ndarray: 
    """
    Apply Otsu's thresholding method to a grayscale image.

    This function computes the histogram of pixel intensities (0-255) and 
    iteratively evaluates all possible thresholds to find the one that 
    minimizes the within-class variance of pixel intensities. The image 
    is then binarized into two classes: background (0) and foreground (255).

    Parameters
    ----------
    brightnessMatrix : np.ndarray
        A 2D NumPy array representing the grayscale image, where each element 
        is an integer brightness value in the range [0, 255].

    Returns
    -------
    np.ndarray
        A 2D NumPy array of the same shape as `brightnessMatrix`, where 
        pixel values are binarized to either 0 (background) or 255 (foreground) 
        based on the optimal Otsu threshold.
    """
    # Generate an array for which to use later
    brightnessCount = np.zeros(256) 
    withinClassVariance = np.zeros(256) 

    # Store each different brightness level and each occurence of brightness pixels
    intermediateLevel, intermediateCount = np.unique(brightnessMatrix, return_counts=True)

    # Constructs and array with the count of each value, 
    # and if no pixels with that value is found sets the count for that level to 0
    for idx, element in enumerate(intermediateLevel):
        brightnessCount[element] = intermediateCount[idx]

    brightnessLevel = np.arange(0, 256)
    totalAmountPixels = np.sum(brightnessCount)

    for threshold, element in enumerate(brightnessCount):
        # Intermediate Steps
        foregroundSum = np.sum(brightnessCount[threshold:])
        backgroundSum = np.sum(brightnessCount[:threshold])

        # Weights
        foregroundWeight = foregroundSum/totalAmountPixels
        backgroundWeight = 1 - foregroundWeight

        # Foreground mean and variance
        if foregroundSum != 0:
            foregroundMean = np.sum((brightnessLevel[threshold:]*brightnessCount[threshold:]))/foregroundSum
            foregroundVariance = np.sum((((brightnessLevel[threshold:]-foregroundMean)**2)*brightnessCount[threshold:]))/foregroundSum

        else:
            foregroundMean = 0
            foregroundVariance = 0

        # Background mean and variance
        if backgroundSum != 0:
            backgroundMean = np.sum((brightnessLevel[:threshold]*brightnessCount[:threshold]))/backgroundSum
            backgroundVariance = np.sum((((brightnessLevel[:threshold]-backgroundMean)**2)*brightnessCount[:threshold]))/backgroundSum

        else:
            backgroundMean = 0
            backgroundVariance = 0

        withinClassVariance[threshold] = backgroundWeight*backgroundVariance+foregroundWeight*foregroundVariance

        # print(f"""
        #     Intermediate steps:
        #         totalAmountPixels: {totalAmountPixels},
        #         backgroundSum: {backgroundSum},
        #         foregroundSum: {foregroundSum},
        #     Weights:
        #         backgroundWeight: {backgroundWeight},
        #         foregroundWeight: {foregroundWeight},
        #     Mean: 
        #         backgroundMean: {backgroundMean},
        #         foregroundMean: {foregroundMean},
        #     Variance:
        #         backgroundVariance: {backgroundVariance},
        #         foregroundVariance: {foregroundVariance},
        #     Within Class Variance:
        #         WCV: {withinClassVariance[threshold]},
        # """)
        
    thresholdMatrix = np.copy(brightnessMatrix)
    thresholdMatrix[thresholdMatrix < np.argmin(withinClassVariance)] = 0
    thresholdMatrix[thresholdMatrix >= np.argmin(withinClassVariance)] = 255

    return thresholdMatrix

if __name__=="__main__":
    with Image.open("gingerbread.jpg") as img:
        img = img.convert("L")
        brightnessValues = np.array(img)

    histogram(brightnessValues)
    ostu_img = Image.fromarray(otsu(brightnessValues))
    ostu_img.save("gingerbread_otsu.jpg")
    # ostu_img.show()
    
    
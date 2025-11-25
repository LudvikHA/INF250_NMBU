import skimage.filters as skif
import skimage.measure as skime
import skimage.morphology as skim
import skimage.segmentation as skis
import skimage.feature as skife
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

with Image.open(r"Assignment-2\IMG_2754_nonstop_alltogether.JPG") as figure:
    figure = figure.convert("L")
    figure = figure.crop((350, 150, 5200, 3400))
    img = np.array(figure)

# Mean thresholding, otsu didnt return a good result
threshold = skif.threshold_mean(img)
img[img >= threshold] = 255
img[img < threshold] = 0

# Make a kernal for use in noise reduction and clean up
kernal = np.ones((25, 25))

print("Removing noise and cleaning up edges...")
# Remove noise and clean up edges
img = skim.area_closing(img, 13000)
print("Closing is done!")
img = skim.area_opening(img, 12000)
print("Opening is done!")

print("Watershedding...")
distance = ndi.distance_transform_edt(img)
localMaximum = skife.peak_local_max(img, footprint=np.ones((30, 30)), labels=img, min_distance=150)
markers = ndi.label(distance)[0]
labels = skis.watershed(-distance, markers, mask=img, watershed_line=True)
properties = skime.regionprops(labels)

fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)

for regions in properties:
        ax.plot(regions.coords[:, 1], regions.coords[:, 0], linewidth=2)

plt.imshow(labels, cmap="nipy_spectral")
plt.axis("off")
plt.show()








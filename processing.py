import os
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt


def unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def process(filename, output_folder, verbose):
    img = cv.imread(filename)
    grayscale = cv.imread(filename, 0)
    filename = os.path.basename(filename).split('.')[-2]

    # Edge enchancement
    unsharp = unsharp_mask(grayscale, amount=6, threshold=0)
    bilateral_blur = cv.bilateralFilter(unsharp, 9, 75, 75)
    _, th = cv.threshold(bilateral_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Noise reduction
    kernel = np.ones((3,3), np.uint8)
    m_blur = cv.medianBlur(cv.medianBlur(th, 3), 3)
    opening = cv.morphologyEx(m_blur, cv.MORPH_OPEN, kernel, iterations = 1)

    # Watershed segmentation
    sure_bg = cv.dilate(opening, kernel, iterations=2)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 0)
    dist_transform = cv.normalize(dist_transform, dist_transform, 0, 255, cv.NORM_MINMAX)
    _, sure_fg = cv.threshold(dist_transform, 0.23 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg * 100000)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    markers = cv.watershed(img, markers)
    bordered = opening
    bordered[markers == -1] = 0
    bordered = cv.convertScaleAbs(bordered)
    bordered = cv.erode(bordered, kernel, iterations=1)

    # Contour detection
    contours, hierarchy = cv.findContours(bordered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv.contourArea(c)>5]  # Filtering erroneous contours (with tiny areas)
    moments = [cv.moments(c) for c in contours]
    centroids = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in moments]

    # Scaling contours up (to revert errosion effect)
    contours = [np.asarray(((c - cent)*1.05)+cent, int) for c, cent in zip(contours, centroids)]

    result = cv.cvtColor(grayscale, cv.COLOR_GRAY2RGB)
    cv.drawContours(result, contours, -1, (0,255,0), thickness=1, lineType=cv.LINE_AA)

    contours_filled = np.zeros(grayscale.shape)
    cv.fillPoly(contours_filled, pts=contours, color=(255,255,255))

    # Calculation of equivalent diameter
    equivalent_diameters = [np.sqrt(4 * m['m00'] / np.pi) for m in moments]
    max_ = max(equivalent_diameters)
    min_ = min(equivalent_diameters)

    contours_colored = cv.cvtColor(grayscale, cv.COLOR_GRAY2RGB)

    for c, d in zip(contours, equivalent_diameters):
        color = (int(255*(d-min_)/(max_-min_)), 0, int(255-255*(d-min_)/(max_-min_)))
        cv.drawContours(contours_colored, [c], -1, color, thickness=2, lineType=cv.LINE_AA)

    colormap = colors.LinearSegmentedColormap.from_list("BuRd", [(0,0,1), (1,0,0)], 256)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.title('Particles colored with respect to equivalent diameter.')
    plt.imshow(contours_colored, colormap)

    cbar = plt.colorbar(fraction=0.0455, pad=0.04)
    interp = lambda x: (int(x)*(max_-min_)+255*min_)//255
    cbar.ax.set_yticklabels([interp(l.get_text()) for l in cbar.ax.get_yticklabels()])

    plt.savefig(output_folder+filename+'_contours_colored.png', dpi=200)

    # Size distribution
    plt.figure(figsize=(10,3))
    plt.title('Particle size distribution.')
    nums, bins, patches = plt.hist(equivalent_diameters, bins=16)
    bins /= max(bins)

    for b, p in zip(bins, patches):
        plt.setp(p, 'facecolor', colormap(b))

    plt.savefig(output_folder+filename+'_size_distribution.png', dpi=150)

    cv.imwrite(output_folder+filename+'_contours_filled.png', contours_filled)
    if verbose:
        cv.imwrite(output_folder+filename+'_unsharp.png', unsharp)
        cv.imwrite(output_folder+filename+'_th.png', th)
        cv.imwrite(output_folder+filename+'_opening.png', opening)
        cv.imwrite(output_folder+filename+'_sure_bg.png', sure_bg)
        cv.imwrite(output_folder+filename+'_dist_transform.png', dist_transform)
        cv.imwrite(output_folder+filename+'_sure_fg.png', sure_fg)
        cv.imwrite(output_folder+filename+'_unknown.png', unknown)
        cv.imwrite(output_folder+filename+'_markers.png', markers)
        cv.imwrite(output_folder+filename+'_contours.png', result)


def main(filenames, output_folder, verbose):
    if output_folder is None:
        output_folder = './results/'
    if output_folder[-1] != '/':
        output_folder+='/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in filenames:
        process(filename, output_folder, verbose)

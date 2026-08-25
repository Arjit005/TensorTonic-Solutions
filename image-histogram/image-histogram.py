def image_histogram(image: list) -> list:
    """
    Returns a list of intensity and count pairs.
    """
    # Write code here
    #A histogram knows "how many" pixels have certain values, but not "where" those pixels are.

    """
    Image
     ↓
    Histogram / HOG / SIFT / other features
     ↓
    Feature vector
     ↓
    SVM / Random Forest / XGBoost
    """
   
    hist = [0] * 256

    # Count every pixel
    for row in image:
        for pixel in row:
            hist[pixel] += 1

    # Build sparse result
    result = []

    for intensity in range(256):
        if hist[intensity] > 0:
            result.append([intensity, hist[intensity]])

    return result
    
MAX_DEPTH = 120.0
MIN_DEPTH = 0.0

def predict(model, images, minDepth=MIN_DEPTH, maxDepth=MAX_DEPTH, batch_size=8):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

predictions = predict(model,test_data)

# if you want to show them using imshow (get rid of 4th dimension)
new_pred = predictions[:,:,:,0]
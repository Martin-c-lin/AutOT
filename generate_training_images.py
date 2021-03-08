import numpy as np
import os
import matplotlib.pyplot as plt

def generate_target_image(data, target_radius=50):
    s = np.shape(data[0])
    target = np.zeros(s)
    Y, X = np.ogrid[:s[0], :s[1]]
    for tx, ty in zip(data[1], data[2]):
        dist_from_center = np.sqrt((X - tx)**2 + (Y - ty)**2)
        target[dist_from_center<target_radius] = 1
    return target

def generate_target_image_set(data, target_shape, nbr_images=10, width=400):

    image_shape = np.shape(data[0])
    #print(target_shape, target_shape[0])
    # Check that target_shape is ok
    if target_shape[0] > image_shape[0] or target_shape[1] > image_shape[1]:
        raise Exception('Target images must not be larger than original images!')

    # Generate target data and arrays
    main_target = generate_target_image(data)
    small_targets = np.zeros([nbr_images, target_shape[0], target_shape[1]])
    small_images = np.zeros([nbr_images, target_shape[0], target_shape[1]])

    # Check which parts of the images are of largest interest
    # Area of interest is uniformly distributed with center in mean position
    # of the particles
    if len(data[1]) > 0:
        xmin = max(int(np.mean(data[1]) - width - target_shape[0]/2), 0)
        ymin = max(int(np.mean(data[2]) - width - target_shape[1]/2), 0)
        polymerized_areas = True
    else:
        xmin = int(image_shape[0]/2 - width)
        ymin = int(image_shape[1]/2 - width)
        polymerized_areas = False

    #print('min values', xmin, ymin, np.mean(data[1]), np.mean(data[2]), image_shape)
    for i in range(nbr_images):
        if polymerized_areas:
            x = min(np.random.randint(xmin, xmin+2*width), image_shape[0]-target_shape[0])
            y = min(np.random.randint(ymin, ymin+2*width), image_shape[1]-target_shape[1])
        else:
            y = np.random.randint(0, image_shape[0]-target_shape[0])
            x = np.random.randint(0, image_shape[1]-target_shape[1])
            #print('x, y ' ,x, y, np.shape(small_targets[i,:,:]), target_shape[0], target_shape[1])
        small_targets[i,:,:] = main_target[y:y+target_shape[1], x:x+target_shape[0]]
        small_images[i,:,:] = data[0][y:y+target_shape[1], x:x+target_shape[0]]

    return small_images, small_targets

def get_training_data(data_path, target_shape, nbr_subimages=10, width=400, max_images=100):

    file_names = [f for f in os.listdir(data_path) if f[:13]=='training_data']
    max_images = min(len(file_names), max_images)
    # Create np arrays to save the data in
    total_nbr_images = int(max_images * nbr_subimages)
    images = np.uint8(np.zeros((total_nbr_images, target_shape[0], target_shape[1])))
    targets = np.uint8(np.zeros((total_nbr_images, target_shape[0], target_shape[1])))

    for idx, file_name in enumerate(file_names[:max_images]):
        data = np.load(data_path+file_name, allow_pickle=True)
        mini_images, mini_targets = generate_target_image_set(data,
                                            target_shape, nbr_subimages, width)
        images[idx*10:(idx+1)*10,:,:] = mini_images
        targets[idx*10:(idx+1)*10,:,:] = mini_targets
        print(idx)
    return images, targets

def check_training_data(folder, new_folder):
    # Function for checking if the data is ok
    try:
        os.makedir(new_folder)
    except:
        print('Folder already exist')
    for file in os.listdir(folder):
        if file[:13] == 'training_data':
            data = np.load(path+file, allow_pickle=True)
            plt.imshow(data[0])
            plt.plot(data[1], data[2],'*r')
            plt.title(file)
            plt.show()

            ok = input('Is file ok')
            if ok == 'NO':
                print('Bad file, will not be moved')
            else:
                print('Good file, will be saved in new folder')
                np.save(new_folder+file, data, allow_pickle=True)
            plt.clf()


path = 'D:/TestTrainingData/'
files = os.listdir(path)
images, targets = get_training_data(path,(1000, 1000))
'''
print(relevant_files)
data_1 = np.load(path+files[7], allow_pickle=True)

images, targets = generate_target_image_set(data=data_1, target_shape=(1000, 1000) )
'''
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
print(np.shape(images))
for image, target in zip(images[-10:], targets[-10:]):
    ax1.imshow(image)
    #ax1.plot(data_1[1], data_1[2], '*r')
    ax2.imshow(target)
    plt.pause(1)

'''
target = generate_target_image(data_1)

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(data_1[0])
ax1.plot(data_1[1], data_1[2], '*r')
ax2.imshow(target)

# plt.imshow(data_1[0])
# plt.plot(data_1[1], data_1[2], '*r')
plt.show()
'''

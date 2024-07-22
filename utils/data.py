import numpy as np

def load_sc_data(img_downsample = 1,
                 use_sensorium_normalization = False,
                 use_zscore_normalization = True,
                 data_path = 'data/sc_processed_data.npz'):

    dat = np.load(data_path, allow_pickle=True)
    img_train = dat['images_train'].astype(np.float32)
    img_val = dat['images_val'].astype(np.float32)
    img_test = dat['images_test'].astype(np.float32)
    spks_train = dat['spks_train'].astype(np.float32)
    spks_val = dat['spks_val'].astype(np.float32)
    spks_test_rep = dat['spks_test'].astype(np.float32) 
    fev_all = dat['fev']


    print(img_train.shape, img_val.shape, img_test.shape)
    print(spks_train.shape, spks_val.shape, spks_test_rep.shape)
    print(fev_all.shape)

    print('downsample images:')
    import cv2
    img_train = np.array([cv2.resize(img, (img.shape[1]//img_downsample, img.shape[0]//img_downsample)) for img in img_train])
    img_val = np.array([cv2.resize(img, (img.shape[1]//img_downsample, img.shape[0]//img_downsample)) for img in img_val])
    img_test = np.array([cv2.resize(img, (img.shape[1]//img_downsample, img.shape[0]//img_downsample)) for img in img_test])
    print('train images:', img_train.shape)
    print('val images:', img_val.shape)
    print('test images:', img_test.shape)

    img_all = np.concatenate([img_train, img_val, img_test], axis=0)
    img_mean = img_all.mean()
    img_std = img_all.std()
    img_train = (img_train - img_mean) / img_std
    img_val = (img_val - img_mean) / img_std
    img_test = (img_test - img_mean) / img_std

    if use_sensorium_normalization:
        spks_std = spks_train.std(axis=1)
        spks_train = spks_train / spks_std[:, None]
        spks_val = spks_val / spks_std[:, None]
        spks_test_rep = spks_test_rep / spks_std[:, None]
    elif use_zscore_normalization:
        spks_std = spks_train.std(axis=1)
        spks_mean = spks_train.mean(axis=1)
        spks_train = (spks_train - spks_mean[:, None]) / spks_std[:, None]
        spks_val = (spks_val - spks_mean[:, None]) / spks_std[:, None]
        spks_test_rep = (spks_test_rep - spks_mean[:, None]) / spks_std[:, None]
    return img_train, img_val, img_test, spks_train, spks_val, spks_test_rep, fev_all
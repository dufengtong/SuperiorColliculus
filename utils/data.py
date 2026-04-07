import numpy as np
import csv
import os

dbs = [{}, {}, {}, {}, {}]

dbs[0] = {'mname': 'SS126', 'datexp': '2024-07-31'}
dbs[1] = {'mname': 'SS126', 'datexp': '2024-09-17'}
dbs[2] = {'mname': 'SS127', 'datexp': '2024-09-04'}
dbs[3] = {'mname': 'SS127', 'datexp': '2024-10-02'}
dbs[4] = {'mname': 'all', 'datexp': 'data'}

import numpy as np

def zscore_nan(spks):
    """
    Z-scores each neuron (each row) in the given 2D array, handling NaN values properly.
    
    Parameters:
    spks (numpy.ndarray): 2D array with shape (NN, NT), where NN is the number of neurons
                          and NT is the number of time points. It may contain NaN values.
    
    Returns:
    numpy.ndarray: Z-scored 2D array with the same shape as input.
    """
    # Create a copy to avoid modifying the original input
    zscored_spks = np.empty_like(spks)
    
    for i in range(spks.shape[0]):  # Iterate over each neuron
        # Extract the current neuron data and ignore NaNs for mean and std calculation
        neuron_data = spks[i, :]
        mean = np.nanmean(neuron_data)
        std = np.nanstd(neuron_data)
        
        # Z-score with NaN handling
        if std > 0:  # Avoid division by zero
            zscored_spks[i, :] = (neuron_data - mean) / std
        else:
            zscored_spks[i, :] = 0  # Set all-zero row if std is zero

    return zscored_spks


def extract_number_from_path(file_path):
    # Split the path by backslashes and get the last part (filename)
    filename = file_path.split('\\')[-1]
    # Remove the file extension and convert to integer
    number = int(filename.split('.')[0])
    return number

def read_and_parse_csv(file_path):
    numbers = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if '.jpg' in row[0]:  
                number = extract_number_from_path(row[0])
                numbers.append(number)
    numbers = np.array(numbers)
    return numbers

def load_sc_data(img_downsample = 1,
                 use_sensorium_normalization = False,
                 use_zscore_normalization = True,
                 data_path = 'data/sc_processed_data.npz'):

    dat = np.load(data_path, allow_pickle=True)
    images = dat['images'].astype(np.float32)
    spks_test_rep = dat['spks_test'].astype(np.float32)
    spks_train = dat['spks_train'].astype(np.float32)
    train_istim = dat['train_istim']
    test_istim = dat['test_istim']
    fev_all = dat['fev']

    itrain = np.arange(len(train_istim))
    np.random.seed(0)
    np.random.shuffle(itrain)
    n_val = int(0.2 * len(itrain))
    ival = itrain[:n_val]
    itrain = itrain[n_val:]
    print(itrain.shape, ival.shape)

    img_test = images[test_istim]
    img_train = images[train_istim[itrain]]
    img_val = images[train_istim[ival]]
    spks_val = spks_train[:, ival]
    spks_train = spks_train[:, itrain]

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
        spks_std = np.nanstd(spks_train, axis=1)
        spks_train = spks_train / spks_std[:, None]
        spks_val = spks_val / spks_std[:, None]
        spks_test_rep = spks_test_rep / spks_std[:, None]
    elif use_zscore_normalization:
        spks_std = np.nanstd(spks_train, axis=1)
        spks_mean = np.nanmean(spks_train, axis=1)
        spks_train = (spks_train - spks_mean[:, None]) / spks_std[:, None]
        spks_val = (spks_val - spks_mean[:, None]) / spks_std[:, None]
        spks_test_rep = (spks_test_rep - spks_mean[:, None]) / spks_std[:, None]
    return img_train, img_val, img_test, spks_train, spks_val, spks_test_rep, fev_all

def load_sc_data_version1(img_downsample = 1,
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

def load_dataset(root, iplanes = None, deconv = 0, subcells = True): #, nsub = 1):    
    from suite2p.extraction import dcnv

    # mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    # root = os.path.join('/home/carsen/dm11_pachitariu/data/PROC', mname, datexp, blk)
    dat_dir = os.path.join(root, 'suite2p')
    ops = np.load(os.path.join(dat_dir, 'ops.npy'), allow_pickle=True).item() # ops is a "pickled" object,

    if iplanes is None:
        iplanes = np.arange(ops['nplanes'])
    # loop over planes and concatenate
    
    # for n in iplanes: 
    ops = np.load(os.path.join(dat_dir, 'ops.npy'), allow_pickle=True).item()

    stat = np.load(os.path.join(dat_dir, 'stat.npy'), allow_pickle=True)        
    
    F0 = np.load(os.path.join(dat_dir, 'F.npy'))
    Fneu0 = np.load(os.path.join(dat_dir, 'Fneu.npy'))
    
    F0 = F0 - 0.7*Fneu0
    #F0 = Fneu0

    
    if subcells:
        icell = np.load(os.path.join(dat_dir, 'iscell.npy'))[:,0]
    else:            
        icell = np.ones((len(F0),), 'bool')

    ypos0 = np.array([stat[n]['med'][0] for n in range(len(stat))])[icell>.5] 
    xpos0 = np.array([stat[n]['med'][1] for n in range(len(stat))])[icell>.5] 
    
    if 'dy' in ops:
        ypos0 += ops['dy'] # add the per plane offsets (dy,dx)
        xpos0 += ops['dx'] # add the per plane offsets (dy,dx)


    F0 = F0[icell>.5]

    #F0 = F0[:, ::nsub]

    Fn = dcnv.preprocess(F0, ops['baseline'], 2 * ops['win_baseline'],
                            ops['sig_baseline'], ops['fs'] , ops['prctile_baseline']) # / nsub

    Fb = F0 - Fn
    #F0 = (F0 - Fb) / np.maximum(10, Fb)
    F0 = (F0 - Fb) / np.maximum(10, Fb.mean())

    if deconv:
        F0     = dcnv.oasis(F0, ops['batch_size'], ops['tau'], ops['fs']) #/ nsub

    F = F0
    xpos = xpos0
    ypos = ypos0
    iplane = np.zeros_like(xpos0)
        # print('plane %d, '%n, 'neurons: %d'%F0.shape[0])

    print('total neurons %d'%F.shape[0])
    return F, ops, xpos, ypos, iplane
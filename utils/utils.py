import approxineuro.encoder_multiconv as encoder

def build_model(n_layers, n_conv, n_conv_mid,highres=True, multikernel=False, multikernel_sizes=[9,25,49], 
                depth_separable=True, pool=True, clamp=True, use_sensorium_normalization=True):
    if use_sensorium_normalization: loss_function = 'poisson'
    else: loss_function = 'mse'

    dense = False
    if pool: stride = 1
    else: stride = 2 if highres else 1
    in_channels = [1, n_conv]
    # in_channels = [1, 16, 32, 64, 128, 320]
    # in_channels = in_channels[:(n_layers+1)]
    kernel_size = [25, 9] if highres else [9, 9]

    for n in range(1, n_layers):
        in_channels.append(n_conv_mid)
        kernel_size.append(5)

    neurons_params = None       
    core = encoder.Core(in_channels, kernel_size, stride, 
                        dense=dense, multikernel=multikernel, kernel_size_conv1=multikernel_sizes, \
                        depth_separable=depth_separable, pool=pool)#[::2])

    in_shape = (in_channels[-1], input_Ly//(2 if highres else 1), input_Lx//(2 if highres else 1))
    # in_shape = (in_channels[-1], input_Ly, input_Lx)
    print('input shape of readout: ', in_shape)

    readout = encoder.Readout(in_shape, len(ineur), rank=1,
                                    yx_separable=True, sigma=0., bias_init=None, poisson=use_sensorium_normalization) #-spks_mean[ineur])

    model = encoder.Encoder(core, readout, loss_fun=loss_function).to(device)
    return model, in_channels

def create_model_name(weight_path, in_channels, fev_threshold, highres=True, multikernel=False, multikernel_sizes=[9,25,49], 
                      depth_separable=True, pool=True, clamp=True, use_sensorium_normalization=True, gabor=False, 
                      rank=1, orth_reg=0, highvar=False, use_30k=False, l1_readout=0.0):
    n_layers = len(in_channels) - 1
    model_save_name = os.path.join(weight_path, f'sc_{n_layers}layer_fev{fev_threshold}')
    for nc in in_channels[1:]:
        model_save_name += f'_{nc}'
    if clamp:
        model_save_name += '_clamp'
    if multikernel:
        model_save_name += '_multikernel'
        for k in multikernel_sizes:
            model_save_name += f'_{k}'
    if use_sensorium_normalization:
        model_save_name += '_sensorium'
    if gabor:
        model_save_name += '_gabor'
    if rank > 1:
        model_save_name += f'_Wcrank{rank}'
    if depth_separable:
        model_save_name += '_depthsep'
    if pool:
        model_save_name += '_pool'
    if orth_reg > 0:
        model_save_name += f'_orth{orth_reg}'
    if highvar:
        model_save_name += '_highvar'
    if use_30k:
        model_save_name += '_30k'
    if l1_readout > 0:
        model_save_name += f'_l1readout{l1_readout}'
    model_path = model_save_name + '.pt'
    return model_path
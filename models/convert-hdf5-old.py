import sys
import os
import h5py
import numpy as np
import json

directory = '.'

for filename in os.listdir(directory):
    if not filename.endswith('.old.hdf5'): continue
    print filename

    hdf5_file = h5py.File(os.path.join(directory, filename), mode='r')

    model_name, file_extension = os.path.splitext(filename)

    if 'layer_names' not in hdf5_file.attrs and 'model_weights' in hdf5_file:
        f = hdf5_file['model_weights']
    else:
        f = hdf5_file
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]


    if not os.path.exists(model_name): os.mkdir(model_name)


    for layer_name in layer_names:
        g = f[layer_name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]

        print layer_name

        for weight_name in weight_names:
            print weight_name

        if layer_name.startswith('lstm_'):
            # kernel = g[layer_name + '/kernel:0'].value
            # recurrent_kernel = g[layer_name + '/recurrent_kernel:0'].value
            # bias = g[layer_name + '/bias:0'].value

            # units = recurrent_kernel.shape[0]


            # kernel_i = kernel[:, :units]
            # kernel_f = kernel[:, units: units * 2]
            # kernel_c = kernel[:, units * 2: units * 3]
            # kernel_o = kernel[:, units * 3:]

            # recurrent_kernel_i = recurrent_kernel[:, :units]
            # recurrent_kernel_f = recurrent_kernel[:, units: units * 2]
            # recurrent_kernel_c = recurrent_kernel[:, units * 2: units * 3]
            # recurrent_kernel_o = recurrent_kernel[:, units * 3:]

            # bias_i = bias[:units]
            # bias_f = bias[units: units * 2]
            # bias_c = bias[units * 2: units * 3]
            # bias_o = bias[units * 3:]

            recurrent_kernel_o = g[layer_name + '_U_o:0'].value
            recurrent_kernel_c = g[layer_name + '_U_c:0'].value
            recurrent_kernel_f = g[layer_name + '_U_f:0'].value
            recurrent_kernel_i = g[layer_name + '_U_i:0'].value

            kernel_o = g[layer_name + '_W_o:0'].value
            kernel_c = g[layer_name + '_W_c:0'].value
            kernel_f = g[layer_name + '_W_f:0'].value
            kernel_i = g[layer_name + '_W_i:0'].value

            # print "--", layer_name + '_b_o:0'
            bias_o = g[layer_name + '_b_o:0'].value
            bias_c = g[layer_name + '_b_c:0'].value
            bias_f = g[layer_name + '_b_f:0'].value
            bias_i = g[layer_name + '_b_i:0'].value


            # Ns = units
            # Ni = kernel.shape[0]
            Ni, Ns = kernel_f.shape

            print Ni, Ns

            W = np.zeros((Ns, Ns + Ni + 1, 4))
            for i in range(Ns):
                # set the biases
                W[i, 0, 0] = bias_f[i]
                W[i, 0, 1] = bias_i[i]
                W[i, 0, 2] = bias_o[i]
                W[i, 0, 3] = bias_c[i]

                # forget gate weights
                W[i, :, 0][1: 1 + Ni] = kernel_f[:, i]
                W[i, :, 0][1 + Ni:] = recurrent_kernel_f[:, i]

                # input gate weights
                W[i, :, 1][1:1 + Ni] = kernel_i[:, i]
                W[i, :, 1][1 + Ni:] = recurrent_kernel_i[:, i]

                # output gate weights
                W[i, :, 2][1:1 + Ni] = kernel_o[:, i]
                W[i, :, 2][1 + Ni:] = recurrent_kernel_o[:, i]

                # state update weights
                W[i, :, 3][1:1 + Ni] = kernel_c[:, i]
                W[i, :, 3][1 + Ni:] = recurrent_kernel_c[:, i]


            weight_value = W
            bytearr = weight_value.astype(np.float32)#.tobytes()
            shape = 'x'.join(str(x) for x in list(weight_value.shape))
            weight = '_combined'
            name = layer_name  + weight + '-' + shape
            bytearr.tofile(os.path.join(directory, model_name, name))

            # for weight_name in weight_names:
            #     weight_value = g[weight_name].value
            #     bytearr = weight_value.astype(np.float32)#.tobytes()

            #     shape = 'x'.join(str(x) for x in list(weight_value.shape))
            #     weight = weight_name[len(layer_name)+1:].split(":")[0]
            #     name = layer_name + '-weights-' + weight + '-' + shape

            #     bytearr.tofile(os.path.join(directory, model_name, name))

        else:
            for weight_name in weight_names:
                weight_value = g[weight_name].value
                bytearr = weight_value.astype(np.float32)#.tobytes()

                shape = 'x'.join(str(x) for x in list(weight_value.shape))
                weight = weight_name[len(layer_name)+1:].split(":")[0]

                weight = weight.replace('b', 'bias')
                weight = weight.replace('W', 'kernel')

                name = layer_name + '-weights-' + weight + '-' + shape

                bytearr.tofile(os.path.join(directory, model_name, name))


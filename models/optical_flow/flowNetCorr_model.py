def get_correlation_layer(input_l,input_r,max_displacement=20,stride2=2,height= 9,width=28):
    conv_activation = lambda x: activations.relu(x,alpha=0.1) # Use the activation from the FlowNetC Caffe implementation
    conv1_l = Conv2D(64,(7,7), strides=(2,2), padding = 'same', name = 'conv1_l', activation=conv_activation)(input_l)
    conv2_l = Conv2D(128, (5, 5),  strides=(2,2),padding = 'same', name='conv2_l', activation=conv_activation)(conv1_l)
    conv3_l = Conv2D(256, (5, 5),  strides=(2,2),padding = 'same', name='conv3_l', activation=conv_activation)(conv2_l)
    conv1_r = Conv2D(64,(7,7), strides=(2,2), padding = 'same', name = 'conv1_r', activation=conv_activation)(input_r)
    conv2_r = Conv2D(128, (5, 5),  strides=(2,2),padding = 'same', name='conv2_r', activation=conv_activation)(conv1_r)
    conv3_r = Conv2D(256, (5, 5),  strides=(2,2),padding = 'same', name='conv3_r', activation=conv_activation)(conv2_r)
    
    output = []
    for i in range(-max_displacement + 1, max_displacement,stride2):
        for j in range(-max_displacement + 1, max_displacement,stride2):
            padded_a = pad(conv3_l, [[0,0], [0, abs(i)], [0, abs(j)], [0, 0]])
            padded_b = pad(conv3_r, [[0, 0], [abs(i), 0], [abs(j), 0], [0, 0]])
            m = padded_a * padded_b

            height_start_idx = 0 if i <= 0 else i
            height_end_idx = height_start_idx + 9
            width_start_idx = 0 if j <= 0 else j
            width_end_idx = width_start_idx + 28
            cut = m[:, height_start_idx:height_end_idx, width_start_idx:width_end_idx, :]

            final = reduce_sum(cut, axis=3)
            output.append(final)
    corr = stack(output, axis=3,name = 'corr_layer')
    
    conv3_l_redir = Conv2D(32,(3,3),padding = "same",name="conv_redir",activation=conv_activation)(conv3_l)
    concatenated_correlation = concatenate([conv3_l_redir,corr],axis = -1,name="concatenated_correlation")
    
    return concatenated_correlation


def flowNetCorr_model(N_img_height=9, N_img_width=28, N_img_channels=3):
    
    model = tf.Sequential()
    
    model.add(Lambda(lambda x: x, input_shape = (N_img_height,N_img_width,N_img_channels)))
    
    model.add(Conv2D(256,(3,3),
                   strides=(2,2),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='conv3'))
    model.add(ELU())
    
    model.add(Conv2D(512,(3,3),
                   strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv4'))
    model.add(ELU())
    
    model.add(Conv2D(512,(3,3),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv4_1'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
    model.add(Conv2D(512,(3,3),
                   strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv5'))
    model.add(ELU())
    
    model.add(Conv2D(512,(3,3),
                     strides=(1,1),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv5_1'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
    model.add(Conv2D(1024,(3,3),
                   strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv6'))
    model.add(ELU())
    
    model.add(Conv2D(1024,(3,3),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv6_1'))
    model.add(ELU())
    
    model.add(Conv2D(2,(3,3),padding='same',name='predict_flow'))
    model.add(ELU())
    model.add(Flatten(name = 'flatten'))
    
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer = adam, loss = 'mean_squared_error')
    
    
    return model
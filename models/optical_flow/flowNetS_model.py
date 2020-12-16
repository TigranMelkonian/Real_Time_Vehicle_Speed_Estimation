def flowNetS_model(N_img_height = 66, N_img_width = 220, N_img_channels = 3):
    inputShape = (N_img_height, N_img_width, N_img_channels)
    model = tf.Sequential()
    
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = inputShape))

    model.add(Conv2D(64,(3,3),
                   padding='same',
                   kernel_initializer = 'he_normal',
                   name='conv0'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
    model.add(Conv2D(64,(3,3),strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv1'))
    model.add(ELU())
    
    model.add(Conv2D(128,(3,3),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv1_1'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
    model.add(Conv2D(128,(3,3),
                   strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv2'))
    model.add(ELU())
    
    model.add(Conv2D(128,(3,3),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv2_1'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
    model.add(Conv2D(256,(3,3),
                   strides=(2,2),
                   padding='valid',
                   kernel_initializer = 'he_normal',
                   name='conv3'))
    model.add(ELU())
    
    model.add(Conv2D(256,(3,3),
                     padding='same',
                     kernel_initializer = 'he_normal',
                     name='conv3_1'))
    model.add(ELU())
    model.add(ZeroPadding2D())
    
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
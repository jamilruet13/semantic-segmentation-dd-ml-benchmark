from keras import optimizers, metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from libs import datasets_keras
from libs.config import LABELMAP
from libs.util_keras import FBeta
import numpy as np

import wandb
from wandb.keras import WandbCallback

def train_model(dataset, model):
    epochs = 50
#     epochs = 0
    lr     = 1e-3
    size   = 300
    wd     = 1e-2
    bs     = 4 # reduce this if you are running out of GPU memory
    pretrained = True

    config = {
        'epochs' : epochs,
        'lr' : lr,
        'size' : size,
        'wd' : wd,
        'bs' : bs,
        'pretrained' : pretrained,
    }

    wandb.config.update(config)
    checkpointer = ModelCheckpoint('model-resnet50.h5', verbose=1, save_best_only=True)
    # # Define IoU metric
    # def mean_iou(y_true, y_pred):
    #   prec = []
    #   for t in np.arange(0.5, 1.0, 0.05):
    #     y_pred_ = tf.to_int32(y_pred > t)
    #     score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    #     K.get_session().run(tf.local_variables_initializer())
    #     with tf.control_dependencies([up_opt]):
    #         score = tf.identity(score)
    #     prec.append(score)
    # return K.mean(K.stack(prec), axis=0)

    earlystopper = EarlyStopping(patience=5, verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss='categorical_crossentropy',
        metrics=[
            metrics.Precision(top_k=1, name='precision'),
            metrics.Recall(top_k=1, name='recall'),
            FBeta(name='f_beta')
        ]
    )
    
    train_data, valid_data = datasets_keras.load_dataset(dataset, bs)
    _, ex_data = datasets_keras.load_dataset(dataset, 10)
    model.fit_generator(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=[earlystopper,learning_rate_reduction,checkpointer,
        WandbCallback(
                input_type='image',
                output_type='segmentation_mask',
                validation_data=ex_data[0]
            )
        ]
    )
  

import os
from model_baseline import *
from data import *

DATASET_NAME = 'eye'
TRAINED_MODEL_EPOCH = None
EPOCH_START = 1
EPOCH_END = 51
STEPS_PER_EPOCH = 500

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/' + DATASET_NAME + '/train','image','label',data_gen_args,save_to_dir = None, image_color_mode='rgb')
model = unet() # new model
if TRAINED_MODEL_EPOCH:
    trained_weights_name = "unet_" + DATASET_NAME + "-baseline-%d.hdf5" % (TRAINED_MODEL_EPOCH)
else :
    trained_weights_name = None
model = unet(pretrained_weights=trained_weights_name) # load pretrained model
#  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#  model_file = "unet_" + DATASET_NAME
#  model_file += "-{epoch:02d}-{val_acc:.2f}.hdf5"
#  model_file += "-{i:02d}.hdf5"

test_set_dir = "data/" + DATASET_NAME + "/test"
predicted_set_dir = "data/" + DATASET_NAME + "/test-predicted-rgb-baseline"
if not os.path.exists(predicted_set_dir):
    os.makedirs(predicted_set_dir)

for i in range(EPOCH_START, EPOCH_END):
    new_model_file = "unet_eye-baseline-" + str(i) + ".hdf5"
    model_checkpoint = ModelCheckpoint(filepath=new_model_file, monitor='val_acc', mode='auto', verbose=1, save_best_only=False, save_weights_only=False, period=1)
    # train the model
    model.fit_generator(myGene,steps_per_epoch=STEPS_PER_EPOCH,epochs=1,callbacks=[model_checkpoint])
    # test the model for each epoch
    testGene = testGenerator(test_set_dir)
    results = model.predict_generator(testGene, 40, verbose=1)
    saveResult(predicted_set_dir, results, epochs=i, steps=STEPS_PER_EPOCH)

#  imgs_train,imgs_mask_train = geneTrainNpy("data/" + DATASET_NAME + "/train/aug/","data/" + DATASET_NAME + "/train/aug/")
#  model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


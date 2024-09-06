base_architecture = 'vgg19'
img_size = 224
prototype_shape = (20, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'server-CBIS-002'

data_path = './../CBIS/images/'
train_dir = data_path + 'training_augmented/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
mask_dir = './../CBIS/masks_augmented/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'msk_crs_ent': 0.1,
    'msk_clst': 0.08,
    'msk_sep': -0.008,
    'fine': 0.000001,
}

num_train_epochs = 100
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

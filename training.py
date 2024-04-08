import fastai.vision.all as f

# get dataset
path = f.untar_data(f.URLs.PETS)

# define dataloader
dls = f.ImageDataLoaders.from_name_re(path, f.get_image_files(path/'images'),
                                      pat='(.+)_\d+.jpg', item_tfms=f.Resize(460),
                                      batch_tfms=f.aug_transforms(size=224, min_scale=0.75))

# train
learn = f.vision_learner(dls, f.models.resnet50, metrics=f.accuracy)
learn.fine_tune(1)

# export
learn.path = f.Path('.')
learn.export()
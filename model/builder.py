
def build_model(model_name, num_classes):
    if model_name == 'FBSNet':
        from model.FBSNet import FBSNet
        return FBSNet(classes=num_classes)
import torch.optim as optim

from models.ResNet3DMedNet import generate_resnet3d
from models.ResNet3Dclassify import generate_resnet_classify
from models.Unet3D import UNet3D
from models.Vnet import VNet
from models.VNET_new import VNet_new
from models.Unet3D_attention import UNet3D_attention
from models.Unet3D_Rattention import UNet3D_Rattention
from models.UARAI import UARAI
from models.Unet3D_Rattention_edgedetect import UNet3D_Rattention_edgedetect
from models.VoxResNet import VoxResNet
from models.UNETR import UNETR
from models.csnet_3d import CSNet3D
from models.ER_Net import ER_Net
from models.csnet_3d import CSNet3D
from models.APAUnet import APAUNet
from models.UARAI_classify import UARAI_classify

model_list = ['csnet3d','ernet','apanet','cs2net','UNETR','VoxResnet','UNET3D', 'UARAI_classify','ResNetclassify', 'UNet3D_attention_inception', 'UNet3D_Rattention_new_icp',
              'UNet3D_Rattention_inception', 'UNet3D_Rattention_edgedetect', 'UNet3D_Zengxq', 'UARAI',
              'UNET3D_Nores', 'UNet_noresidual', 'UNet3D_attention', 'UNet3D_attention_addjuchi',
              'UNet3D_Rattention_juchi', 'UNet3D_Rattention', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3',
              'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET", 'UNet3D_Rattention_new_ioudice',
              'UNet3D_Zengxq_residual', 'UNet3D_Zengxq_RA', 'UNet3D_Zengxq'
              ,"VNET_new"]


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name=='csnet3d':
        model =CSNet3D(classes=2, channels=1)
    elif model_name=='cs2net':
        model =CSNet3D(classes=2, channels=1)
    elif model_name=='ernet':
        model =ER_Net(classes=2, channels=1)
    elif model_name=='apanet':
        model = APAUNet(1, 2).cuda()


    elif model_name == 'VNET_new':
        model = VNet_new(n_channels=1, n_classes=2)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'VoxResnet':
        model=VoxResNet(in_chns=in_channels, feature_chns=64,class_num=num_classes)
    elif model_name == 'UNETR':
        model = UNETR()
    ################################################################
    elif model_name == 'UNet3D_attention':
        model = UNet3D_attention(in_channels=in_channels, n_classes=num_classes, base_n_filter=8,
                                 attention_dsample=(2, 2, 2), nonlocal_mode='concatenation')
    elif model_name == 'UNet3D_Rattention':  #
        model = UNet3D_Rattention(in_channels=in_channels, n_classes=num_classes, base_n_filter=8, agg_input=32)
    elif model_name == 'UARAI':  #
        model = UARAI(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'UARAI_classify':  #
        model = UARAI_classify(in_channels=in_channels, n_classes=num_classes, base_n_filter=16)
    elif model_name == 'UNet3D_Rattention_edgedetect':  #
        model = UNet3D_Rattention_edgedetect(in_channels=in_channels, n_classes=num_classes, base_n_filter=8,
                                             agg_input=32)
    elif model_name == "RESNETMED3D":
        depth = 10
        model = generate_resnet3d(in_channels=in_channels, classes=num_classes, model_depth=depth)
    elif model_name == "ResNetclassify":
        depth = 18
        model = generate_resnet_classify(in_channels=in_channels, classes=num_classes, model_depth=depth)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))



    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer

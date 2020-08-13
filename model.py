import torch.nn as nn
import torch
import os
import sys
import network
import Loss

class model_wrapper(nn.Module):
    def __init__(self):
        super(model_wrapper, self).__init__()

    def initialize(self,opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoint_dir,opt.name)


    def forward(self):
        pass

    def test(self):
        pass
    # not sure this is the eval mode

    def get_current_visuals(self):
        return self.input

    def name(self):
        return 'model_wrapper'

    def get_current_errors(self):
        return {}

    def save_network(self,network, network_label, epoch_label, gpu_ids):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label,network_label)
        # should save at ./checkpoint/self.name/
        save_path = os.path.join(self.save_dir,save_filename)
        #this should be ./checkpoint/self.name/netG/netD
        # remain to adjust (the path)
        torch.save(network.cpu().state_dict(), save_path)

        if gpu_ids>0 and torch.cuda.is_available():
            # move it back to GPU
            network.cuda()

    def load_network(self,network, network_label, epoch_label, save_dir=''):
        if not save_dir:
            return None
            # if under a first train mode. the path didn't pass in here, so just jump out the fun
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)

        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists !' % save_path)
        else:
            network.load_state_dict(torch.load(save_path))
            # torch.load(save_path, map_location=lambda storage, loc: storage)
            # not sure this need or not
            print('%s loading succeed ' % network_label)
            if self.opt.debug:
                for a in torch.load(save_path).items():
                    print(a)


        # remain to update
        # try:
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     network.load_state_dict(pretrained_dict)
        #     if self.opt.verbose:
        #         print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
        # except:
        #     print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
        #     for k, v in pretrained_dict.items():
        #         if v.size() == model_dict[k].size():
        #             model_dict[k] = v
        #
        #     if sys.version_info >= (3,0):
        #         not_initialized = set()
        #     else:
        #         from sets import Set
        #         not_initialized = Set()
        #
        #     for k, v in model_dict.items():
        #         if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
        #             not_initialized.add(k.split('.')[0])
        #
        # print(sorted(not_initialized))
        # network.load_state_dict(model_dict)
        # remain to update



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def update_learning_rate(self):
        pass

"""
load data
pre-process data
gen done
dis done
loss done 
optimizer done
util tool
save model TBD
load model TBD
eval model
forward() 
更新学习率之类的
"""



class SCAR(model_wrapper):
    def __init__(self):
        super(SCAR, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        # gan = true, usegan = TBD, vgg = TBD, D real = true D fake = true
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            # 打包ganloss featureloss vggloss Dfake D real loss 和 flag
            # 如果是ture 就返回这个loss
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    # 暂时不需要

    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        self.netG = network.create_G(opt.input_chan).to(self.device)
        # input_channel ,K = 64 ,downsample_num = 6
        # dpwmsample_num can be add in opt.downsample_num
        if 'train' in self.opt.mode :
            # train mean we definitely need a D and loss/or new loss?
            # opt.mode = continue train / train / load pre train
            # continue train means go on training by from the last epoch
            # train mean first time train
            # load pre train mean do noting , just load the D and G with params

            # in train mode no matter how we need a D except we are in test/eval mode

            self.netD = network.create_D(opt.input_chan*2).to(self.device)
            # cat the input and target into patchGAN
            # so far we only have 3 channel
            # input_channel,K = 64,n_layers = 4

            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            # load model from the checkpoint
            # something like ./checkpoint/modename/10_net_G.pth
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)

            # TBD
            # self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            # TBD
            # we have gan loss and dis fake / real loss and vgg loss
            self.l1loss = nn.L1Loss()
            self.vggloss = Loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            self.TVloss = Loss.TVLoss()
            self.GANloss = Loss.GANLoss(device = self.device,lsgan=opt.lsgan)
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()),lr=1e-4,betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),lr=1e-4,betas=(0.9, 0.999))
            print('---------- Networks initialized -------------')
            print('---------- NET G -------------')
            print(self.netG)
            print('---------- NET D -------------')
            print(self.netD)
            # remain to push the model to cuda if avliable

        elif 'test' in self.opt.mode :

            self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)

        else:
            print('mode error,this would create a empty netG without pretrain params')

    def forward(self,input):
        target_image = input['image'].to(self.device)
        input_image = input['label'].to(self.device)
        generated = self.netG(input_image)

        cat_fake = torch.cat((generated,input_image),1)
        # detach so that when D update not effect G
        # or cat_fake.detach()
        # fakeimage and input_image
        cat_real = torch.cat((target_image,input_image),1)
        # targetimage and input_image

        dis_real = self.netD(cat_real)
        dis_fake = self.netD(cat_fake.detach())
        # get the patch
        loss_real = self.GANloss(dis_real,True)
        loss_fake = self.GANloss(dis_fake,False)
        # get the patchGAN loss
        dis_loss = (loss_real + loss_fake) * 0.5

        gan_loss = self.GANloss(cat_fake,True)
        # fake to the Dis so that optimize our G to generated more natural image
        l1_loss = self.l1loss(target_image,generated) * 100.0
        # the main loss(content loss)
        TV_loss = self.TVloss(generated)
        # regularize the noise
        vgg_loss = self.vggloss(generated,target_image)
        # style loss

        G_loss = gan_loss + TV_loss + vgg_loss + l1_loss
        return {'G_loss':G_loss,
                'G_ganloss':gan_loss,
                'l1_loss':l1_loss,
                'TV_loss':TV_loss,
                'dis_loss':dis_loss,'dis_real':loss_real,'dis_fake':loss_fake,
                'vgg_loss':vgg_loss}

    def save(self,which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)




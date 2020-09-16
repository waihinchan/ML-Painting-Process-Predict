import torch.nn as nn
import torch
import os
import sys
import net
import Loss
import platform
from net.network import init_weights
class model_wrapper(nn.Module):
    def __init__(self):
        super(model_wrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def initialize(self,opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoint_dir,opt.name) if not opt.load_from_drive else '/content/drive/My Drive/'+opt.name

    def forward(self):
        pass

    def test(self):
        pass

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
        torch.save(network.cpu().state_dict(), save_path)
        # save this to the google drive
        if platform.system() == 'Linux':
            drive_root = os.path.join('/content/drive/My Drive', self.opt.name)
            if not os.path.exists(drive_root):
                os.makedirs(drive_root)
            drive_path = os.path.join(drive_root, save_filename)
            torch.save(network.cpu().state_dict(), drive_path)

        if gpu_ids > 0 and torch.cuda.is_available():
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

class SCAR(model_wrapper):
    def __int__(self):
        super(SCAR, self).__int__()

    def initialize(self,opt):
        model_wrapper.initialize(opt)
        input_nc = opt.input_chan * opt.n_past_frames
        if opt.use_abs:
            input_nc += opt.n_past_frames


        # need to calculate the middle input if cat with the feature from the Encoder
        self.netG = net.generator.global_frame_generator(input_channel = input_nc,
                                                         firstK = opt.firstK,
                                                         n_downsample = opt.n_downsample_global,
                                                         n_blocks = opt.n_blocks)
        self.netG.apply(init_weights)

        # in forward we will cat the orginal image and the t-1frame as the input
        # let encoder count the difference feature between the t-1 frame and the orginal image
        # so that the generator can get the feature to generate the next frame image.
        E_input = opt.input_chan * opt.n_past_frames
        self.netE = net.generator.Encoder(E_input,opt.firstK)
        self.netE.apply(init_weights)

        # only when traning need it
        if 'train' in self.opt.mode :
            self.netD_T = net.discriminator.MultiscaleDiscriminator(input_channel = opt.output_channel,
                                                                         k=opt.firstK,n_layers = 3,num_D = 1)
            self.netD_T.apply(init_weights)
            self.netD = net.discriminator.MultiscaleDiscriminator(input_channel = opt.output_channel,
                                                                        k=opt.firstK,n_layers = 3,num_D = 1)
            self.netD.apply(init_weights)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D_frames', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'D_frame', opt.which_epoch, pretrain_path)
            self.load_network(self.netD, 'G', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'E', opt.which_epoch, pretrain_path)

            # loss
            self.l1loss = nn.L1Loss()
            # for real and fake frame
            self.vggloss = Loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            # for style match

            self.GANloss = Loss.GANLoss(device = self.device,lsgan=opt.lsgan)
            # for dis loss

            G_params = list(self.netG.parameters())+ list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(G_params,lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D_T = torch.optim.Adam(list(self.netD_T.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))


        elif 'test' in self.opt.mode:
                self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
                self.load_network(self.netE, 'E', opt.which_epoch, self.save_dir)
        else:
            print('mode error,it would create a empty netG without pretrain params')



def forward(self,input):
        # input is a list of each frame corresponding to the whole process, also the last frame as the result
        target = input['target']
        real_frames = input['frame']
        fake_frames = [] # this is for the generated image
        for current_real_frame,frame_id in enumerate(real_frames,start=0):

            if frame_id == 0:
                # not sure this would working
                blank_image = torch.zeros_like(target)
                prev_fake_frame = blank_image
                prev_real_frame = blank_image
            else:
                prev_fake_frame = fake_frames[-1]
                # pick the last frame which been add in to the list
                prev_real_frame = real_frames[frame_id-1]

            lossinfo = torch.cat((target,prev_fake_frame),dim=1)
            lossfeature = self.netE(lossinfo)
            current_fake_frame = self.netG(lossfeature)
            # not sure the forward info would record if i only record the current frame and the prev frame
            fake_frames.append(current_fake_frame)
            # the current frame would be the prev frame of the next current frame




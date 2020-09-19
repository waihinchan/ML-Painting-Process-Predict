import torch.nn as nn
import torch
import os
import sys
import net.discriminator
import net.generator
import net.loss
import platform
from net.network import init_weights
# fixed
class model_wrapper(nn.Module):

    def __init__(self):
        super(model_wrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device = %s" % self.device)
    def initialize(self,opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoint_dir,opt.name) if not opt.load_from_drive else '/content/drive/My Drive/'+opt.name
        # this is only for using at colab..

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

        #this should be ./checkpoint/self.name/netG_or_netD
        torch.save(network.cpu().state_dict(), save_path)


        # remain to update according to the option using google drive or not
        # save this to the google drive
        # if platform.system() == 'Linux':
        #     drive_root = os.path.join('/content/drive/My Drive', self.opt.name)
        #     if not os.path.exists(drive_root):
        #         os.makedirs(drive_root)
        #     drive_path = os.path.join(drive_root, save_filename)
        #     torch.save(network.cpu().state_dict(), drive_path)

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
# fixed

class SCAR(model_wrapper):
    def __int__(self):
        super(SCAR, self).__int__()

    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        input_nc = opt.input_chan * (opt.n_past_frames+1)
        # blank image and the last frame as the input. if the n_past_frames greater than 1, the blank image will automatily fill the empty slot



        self.netG = net.generator.global_frame_generator(opt = opt,input_channel = input_nc,
                                                         firstK = opt.firstK,
                                                         n_downsample = opt.n_downsample_global,
                                                         n_blocks = opt.n_blocks).to(self.device)

        self.netG.apply(init_weights)


        # E_input use the last current frames output from the generator and the last frames as input
        # E_input = opt.input_chan * 2
        # self.netE = net.generator.Encoder(E_input,opt.firstK).to(self.device)
        # self.netE.apply(init_weights)

        # only when traning need it
        if 'train' in self.opt.mode:

            D_T_input = (opt.output_channel * opt.n_past_frames) * 2
            self.netD_T = net.discriminator.MultiscaleDiscriminator(input_channel = D_T_input,
                                                                         k=opt.firstK,n_layers = 3,num_D = 1).to(self.device)
            self.netD_T.apply(init_weights)


            D_input = opt.output_channel * (opt.n_past_frames+1)
            self.netD = net.discriminator.MultiscaleDiscriminator(input_channel = D_input,
                                                                        k=opt.firstK,n_layers = 3,num_D = 1).to(self.device)
            self.netD.apply(init_weights)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D_frames', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'D_frame', opt.which_epoch, pretrain_path)
            self.load_network(self.netD, 'G', opt.which_epoch, pretrain_path)
            # self.load_network(self.netG, 'E', opt.which_epoch, pretrain_path)

            # loss
            self.l1loss = nn.L1Loss()
            # for real and fake frame
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            # for style match

            self.GANloss = net.loss.GANLoss(device = self.device,lsgan=opt.lsgan)
            # for dis loss

            # G_params = list(self.netG.parameters()) + list(self.netE.parameters())
            G_params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(G_params,lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D_T = torch.optim.Adam(list(self.netD_T.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))


        elif 'test' in self.opt.mode:
                self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
                self.load_network(self.netE, 'E', opt.which_epoch, self.save_dir)
        else:
            print('mode error,it would create a empty netG without pretrain params')


    def forward(self,input):
        # this is the big forward, forward all the frames
        target = input['target'].to(self.device)

        real_frames = input['frames']
        for k in range(0,len(real_frames)):
            real_frames[k] = real_frames[k].to(self.device)

        fake_frames = []
        # this is for the total fake generated frames
        forward_fake_frames = []
        forward_real_frames = []
        # this is for each time how many fake frames pick from the above fake_frames list base on opt.n_past_frames
        # opt.n_past_frames = 3
        for j in range(0, self.opt.n_past_frames):
            fake_frames += [torch.zeros_like(target).to(self.device)]
            forward_real_frames += [torch.zeros_like(target).to(self.device)]

            # first base on the n_past_frames to fill the empty slot frames
        for i in range(0,len(real_frames)):
            """
            # the next time the fake_frames will like [fake_0_frames,blank_images,blank_images]
            # the forward images will like [fake_0_frames,blank_images,blank_images]
            # then still the netG pick the past 3 frames but with one fake_frames generated by itself 
            # rather than all of them are blank_images
            # then the next next time the fake_1_frames will take place in the second slot which the index = 1            
            """
            if i < self.opt.n_past_frames:
                forward_fake_frames = [target]
                forward_fake_frames += fake_frames[0:self.opt.n_past_frames]
                forward_fake_frames_ = fake_frames[0:self.opt.n_past_frames]
                # this will be the [last_frames,blank_images,blank_images...etc]
                input = torch.cat(forward_fake_frames,dim=1)
                # the input is cat of the last frame and the n_past_frames
                # this will be the shape of (1,CH*(n_past_frames+1),W,H)
                G_output =self.netG(input)
                forward_fake_frames[i] = G_output
                # replace out put of the netG(which mean the fake generated frames) to
                # the blank image to the corresponding slot
                forward_real_frames[i] = real_frames[i]
                # the real_frames are always exist. just take place the orginal one
                # so that it can match to the fake_images index
            else:
                forward_fake_frames = [target]
                forward_fake_frames += fake_frames[i-self.opt.n_past_frames:i]

                # find the corresponding slot of the orginal real frames
                forward_real_frames = real_frames[i-self.opt.n_past_frames+1:i+1]
                input = torch.cat(forward_fake_frames, dim=1)
                G_output = self.netG(input)
                # when out of the blank image range, the list is starting extending the length
                fake_frames.append(G_output)
                forward_fake_frames_ = fake_frames[i - self.opt.n_past_frames + 1:i + 1]
            # **** to this step, we have a fake_frames list generated from the generator **** #
            # **** next is to forward the single frames and the n_past_frames to the TWO discriminator
            # not sure to include the current new fake frames or not...

            del forward_fake_frames[0]
            temp_cat_real = forward_fake_frames + [real_frames[i]]
            temp_cat_fake = forward_fake_frames + [G_output]


            cat_real = torch.cat(temp_cat_real,dim=1)

            # cat the fake_forward_frames and the real current frames
            # we can regard the forward_fake_frames as a input label map
            cat_fake = torch.cat(temp_cat_fake, dim=1)
            # cat the fake_forward_frames and the fake current frames
            # get the featrue from netD
            dis_real = self.netD(cat_real.detach())
            """
            # the cat real is consist of a list of fake image from the generator... so...
            # if the forward happend once then backward maybe this is un-necessary.
            # still remain for test
            """

            dis_fake = self.netD(cat_fake.detach())
            # get the featrue from netD

            # the loss of the single frame discriminator
            loss_real = self.GANloss(dis_real, True)
            loss_fake = self.GANloss(dis_fake, False)
            # the loss of the single frame discriminator
            # the loss of the generator from the discriminator
            gan_loss = self.GANloss(cat_fake, True)

            # **** next is n_past_frames to the T_frames discriminator

            temp_cat_T_fake = forward_fake_frames + forward_fake_frames_
            temp_cat_T_real = forward_fake_frames + forward_real_frames
            cat_T_fake = torch.cat(temp_cat_T_fake,dim=1)
            cat_T_real = torch.cat(temp_cat_T_real , dim=1)
            # get the featrue from netD_T
            dis_T_real = self.netD_T(cat_T_real.detach())
            dis_T_fake = self.netD_T(cat_T_fake.detach())
            # get the featrue from netD_T

            # the loss of the single frame discriminator
            loss_T_real = self.GANloss(dis_T_real, True)
            loss_T_fake = self.GANloss(dis_T_fake, False)
            # the loss of the single frame discriminator
            # the loss of the generator from the discriminator
            gan_T_loss = self.GANloss(cat_T_fake, True)*100

            l1_loss = self.l1loss(real_frames[i], G_output) * 100.0
            vgg_loss = self.vggloss(real_frames[i], G_output)
            G_loss = gan_loss + gan_T_loss + l1_loss + vgg_loss
            D_loss = (loss_real+loss_fake)*0.5
            D_T_loss =  (loss_T_fake+loss_T_real)*0.5
            return {
                "G_loss":G_loss,
                "D_loss":D_loss,
                "D_T_loss":D_T_loss,
                "vgg_loss":vgg_loss,
                "l1_loss":l1_loss,
                "gan_loss":gan_loss,
                "gan_T_loss":gan_T_loss
            }

    def update_learning_rate(self, epoch):
        # eporch should pass a value like current_epoch - start_epoch
        lr = self.opt.learningrate * (0.1 ** (epoch // 30))
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_T.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s loss is %s' % (epoch, lr))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD_T, 'D_T', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
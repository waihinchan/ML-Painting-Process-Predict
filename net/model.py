import torch.nn as nn
import torch
import os
import sys
import net.discriminator
import net.generator
import net.loss
import platform
import time
from net.network import init_weights
from utils import fast_check_result
# ******************************************** don't touch here ***********************************************
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
# ******************************************** don't touch here ***********************************************

# remain to update
class SCAR(model_wrapper):
    def __int__(self):
        super(SCAR, self).__int__()

    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        input_nc = opt.input_chan * (opt.n_past_frames+1)



        self.netG = net.generator.global_frame_generator(opt = opt,input_channel = input_nc,
                                                         firstK = opt.firstK,
                                                         n_downsample = opt.n_downsample_global,
                                                         n_blocks = opt.n_blocks).to(self.device)

        self.netG.apply(init_weights)


        # E_input = opt.input_chan * 2
        # self.netE = net.generator.Encoder(E_input,opt.firstK).to(self.device)
        # self.netE.apply(init_weights)
        if opt.generate_first_frame:
            self.netG_first = net.generator.global_frame_generator(opt=opt, input_channel=opt.input_chan,
                                                             firstK=opt.firstK,
                                                             n_downsample=opt.n_downsample_global,
                                                             n_blocks=opt.n_blocks,generate_first_frame=True).to(self.device)

            self.netG_first.apply(init_weights)
        # only when traning need it
        if 'train' in self.opt.mode:
            D_input = opt.output_channel * (opt.n_past_frames+1)
            self.netD = net.discriminator.MultiscaleDiscriminator(input_channel = D_input,
                                                                        k=opt.firstK,n_layers = 3,num_D = 1).to(self.device)
            self.netD.apply(init_weights)

            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            # self.load_network(self.netE, 'E', opt.which_epoch, pretrain_path)
            if self.opt.generate_first_frame:
                self.load_network(self.netG_first, 'G_first', opt.which_epoch, pretrain_path)


            # loss
            self.l1loss = nn.L1Loss()
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            self.GANloss = net.loss.GANLoss(device = self.device,lsgan=opt.lsgan)
            # self.Tvloss = net.loss.TVLoss()

            # G_params = list(self.netG.parameters()) + list(self.netE.parameters())
            G_params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(G_params,lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))
            if opt.generate_first_frame:
                self.optimizer_G_first = torch.optim.Adam(self.netG_first.parameters(),lr=opt.learningrate,betas=(0.9, 0.999))

        elif 'test' in self.opt.mode:
                self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
                # self.load_network(self.netE, 'E', opt.which_epoch, self.save_dir)
                if opt.generate_first_frame:
                    self.load_network(self.netG_first, 'G_first', opt.which_epoch, self.save_dir)

        else:
            print('mode error,it would create a empty netG without pretrain params')


    def forward(self,input):
        target = input['target'].to(self.device)

        real_frames = input['frames'][1:] if self.opt.generate_first_frame else input['frames']
        real_first_frame = input['frames'][0].to(self.device)
        # if generate first frame we start from the second index
        for k in range(0,len(real_frames)):
            real_frames[k] = real_frames[k].to(self.device)

        fake_frames = []

        if self.opt.generate_first_frame:
            first_frame = self.netG_first(target,None)
            # this should return the first sketch frame
        else:
            first_frame = torch.zeros_like(target)

        for j in range(0, self.opt.n_past_frames):
            fake_frames += [first_frame.detach().to(self.device)]

        # init the total loss
        G_loss = 0
        D_loss = 0
        Vgg_loss = 0
        L1_loss = 0
        Gan_loss = 0

        # init the total loss

        for i in range(0,len(real_frames)):
            forward_fake_frames = [target]
            forward_fake_frames += fake_frames[-self.opt.n_past_frames:]
            # if generate first : forward_fake_frames = [target,first_fake_frame]
            # else forward_fake_frames = [target,blank_image]
            input = torch.cat(forward_fake_frames, dim=1)
            G_output = self.netG(input, fake_frames[-1])
            # G output = target+blankimage|first_fake_frame + blankimage|first_fake_frame
            if i < self.opt.n_past_frames:
                fake_frames[i] = G_output
            else:
                fake_frames.append(G_output)

            del forward_fake_frames[0]
            # don't need the target frames anymore

            temp_cat_real = forward_fake_frames + [real_frames[i]]
            temp_cat_fake = forward_fake_frames + [G_output]
            # if we generate first frame, the real_frames[0] = input['frames'][1] forward_fake_frames = first_frame
            # other wise still real_frames[0] = input['frames'][0] forward_fake_frames = blankimages
            cat_real = torch.cat(temp_cat_real,dim=1)
            cat_fake = torch.cat(temp_cat_fake, dim=1)
            dis_real = self.netD(cat_real.detach())
            dis_fake = self.netD(cat_fake.detach())
            # not sure need a detach becaus the one of the element of the cat_real was from the generator itself,
            # if we not detach, when update the D, will effect the G's params(or not? i still confused on the auto-grad function)
            # or we can said when zero_grad D, we can no longer track the grad of the forward_cat_frames, so the optmizer will throw a error?
            # need some test in the future
            # the loss of the single frame discriminator
            loss_real = self.GANloss(dis_real, True)
            loss_fake = self.GANloss(dis_fake, False)
            # the loss of the single frame discriminator
            gan_loss = self.GANloss(cat_fake, True)
            l1_loss = self.l1loss(real_frames[i], G_output) * 100.0
            vgg_loss = self.vggloss(real_frames[i], G_output)
            g_loss = gan_loss  + l1_loss + vgg_loss
            d_loss = (loss_real+loss_fake)*0.5
            # each frame's loss
            Gan_loss += gan_loss
            L1_loss += l1_loss
            Vgg_loss += vgg_loss
            G_loss += g_loss
            D_loss += d_loss

        if self.opt.generate_first_frame:
            # cat real and fake of the first_frame generator are the same like the label map and the ground truth
            cat_fake = torch.cat((target,first_frame),dim=1)
            cat_real = torch.cat((target,real_first_frame),dim=1)
            dis_real = self.netD(cat_real)
            dis_fake = self.netD(cat_fake.detach())
            loss_real = self.GANloss(dis_real, True)
            loss_fake = self.GANloss(dis_fake, False)
            d_loss = (loss_real + loss_fake) * 0.5
            D_loss += d_loss
            firstG_loss = self.l1loss(real_first_frame, first_frame) * 50.0 + self.vggloss(real_first_frame,first_frame)*50 + self.GANloss(cat_fake, True)

        if self.opt.save_result:
            label = str(time.time())
            for k,ouput in enumerate(fake_frames,start=0):
                fast_check_result.imsave(ouput, dir='./result/result_preview/',index= label + str(k))
            if self.opt.generate_first_frame:
                fast_check_result.imsave(first_frame, dir='./result/result_preview/',index= label + "first_frame")

        return {
            "G_loss":G_loss,
            "D_loss":D_loss,
            "vgg_loss":Vgg_loss,
            "l1_loss":L1_loss,
            "gan_loss":Gan_loss,
            "firstG_loss": firstG_loss if self.opt.generate_first_frame else None
        }

    def update_learning_rate(self, epoch):
        # eporch should pass a value like current_epoch - start_epoch
        lr = self.opt.learningrate * (0.1 ** (epoch // 10))
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        if self.opt.generate_first_frame:
            for param_group in self.optimizer_G_first.param_groups:
                param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s loss is %s' % (epoch, lr))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
        if self.opt.generate_first_frame:
            self.save_network(self.netG_first, 'G_first', which_epoch, self.opt.gpu_ids)

# 明日之星！！！
class sketch_style(model_wrapper):
    """
    let's say we have a sketch image without the style, we want to transfrom it into a style sketch
    kind of like spade.. but spade use resblock, for a simply start i use a VAE or a something like U-net..
    """
    def __init__(self):
        super(sketch_style, self).__init__()
    def initialize(self, opt):
        model_wrapper.initialize(self, opt)
        self.netE = net.generator.Encoder(opt.input_chan)
        self.netE.apply(init_weights())
        # encoder for the style
        # generator for the image
        # generator takes z and the input image
    def forward(self):
        pass

    def encode_z(self, real_image):

        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    def generate_fake(self, input_semantics, real_image):
        z = None
        KLD_loss = None
        z, mu, logvar = self.encode_z(real_image)
        KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        fake_image = self.netG(input_semantics, z=z)


        return fake_image, KLD_loss


# this was from the master branch
class single_frame(model_wrapper):
    def __init__(self):
        super(single_frame, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        self.netG = net.generator.pix2pix_generator(opt.input_chan).to(self.device)
        self.netG.apply(init_weights)
        if 'train' in self.opt.mode :
            self.netD = net.discriminator.patchGAN(opt.input_chan*2).to(self.device)
            self.netD.apply(init_weights)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.l1loss = nn.L1Loss()
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            self.TVloss = net.loss.TVLoss()
            self.GANloss = net.loss.GANLoss(device = self.device,lsgan=opt.lsgan)
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()),lr=opt.learningrate,betas=(0.9, 0.999))
            print('---------- Networks initialized -------------')
            print('---------- NET G -------------')
            print(self.netG)
            print('---------- NET D -------------')
            print(self.netD)
        elif 'test' in self.opt.mode :
            self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
        else:
            print('mode error,this would create a empty netG without pretrain params')
    def forward(self,input):
        target_image = input['target'].to(self.device)
        input_image = input['input'].to(self.device)
        generated = self.netG(input_image)
        cat_fake = torch.cat((generated,input_image),1)
        cat_real = torch.cat((target_image,input_image),1)
        dis_real = self.netD(cat_real)
        dis_fake = self.netD(cat_fake.detach())
        loss_real = self.GANloss(dis_real,True)
        loss_fake = self.GANloss(dis_fake,False)
        dis_loss = (loss_real + loss_fake) * 0.5
        gan_loss = self.GANloss(cat_fake,True)
        l1_loss = self.l1loss(target_image,generated) * 100.0
        TV_loss = self.TVloss(generated)
        vgg_loss = self.vggloss(generated,target_image)
        G_loss = gan_loss + TV_loss + vgg_loss + l1_loss
        if self.opt.save_result:
            fast_check_result.imsave(generated[-1,:,:,:],index=time.time(),dir='./result/result_preview/')
            # only save the last one... for saving space
        return {'G_loss':G_loss,
                'G_ganloss':gan_loss,
                'l1_loss':l1_loss,
                'TV_loss':TV_loss,
                'D_loss':dis_loss,'dis_real':loss_real,'dis_fake':loss_fake,
                'vgg_loss':vgg_loss}
    def save(self,which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
    def update_learning_rate(self,epoch):
        lr = self.opt.learningrate * (0.1 ** (epoch // 10))
        for param_group in self.optimizer_D.param_groups:
         param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s loss is %s'%(epoch,lr))
# backup
class global_frame_generator(nn.Module):
    #when bs = 1 BN = IN , we acutlly can use BN as default..
    def __init__(self,opt,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        n_block is the num of the res-block
        n_downsample is the num of the downsample layer
        firstK is the first kernel num of the conv layer
        """
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        # self.generate_first_frame = generate_first_frame
        ngf = firstK

        downsample = []
        downsample += network.c7s1_k(input_channel,ngf)
        # 7 * 7 output 64 CH
        # downsample layer
        temp_K = ngf

        for i in range(1,n_downsample+1):
            ds = network.dk(temp_K,temp_K*2)
            temp_K*=2
            downsample+=ds

        # downsample layer
        # res-block
        for i in range(1,n_blocks+1):
            res = [resnet.ResK(dim=temp_K,padding_type=padding_type,norm_layer=norm_layer)]
            downsample += res
        # res-block
        # upsample layer
        # should cat with the output from the encoder if the encoder using a vector Z
        self.downsample = nn.Sequential(*downsample)
        upsample = []
        for i in range(1,n_downsample+1):
            us = network.uk(input_channels=temp_K,stride=2,N=3,k=int(temp_K/2))
            temp_K = int(temp_K/2)
            upsample += us
        self.upsample = nn.Sequential(*upsample)
        # upsample layer

        last = network.c7s1_k(temp_K,opt.output_channel)
        self.last = nn.Sequential(*last)


    def forward(self,input,prev_frame,freature=None):
        """
        forward 1 big data will forward this many times.
        each "small" forward take the previous frames and (if use the last frames also) as the input
        pass though the init layer and the downsample layer
        then cat with the feature map from the Encoder (like a U-net but the feature are different)
        not sure should combine or not as quite like wasting caculation
        """
        x = input
        x = self.downsample(x)
        # inlcude resblock
        x = self.upsample(x)
        x = self.last(x)
        x = x + prev_frame
        return x


class pix2pix_generator(nn.Module):
    def __init__(self,input_channel ,K = 64 ,downsample_num = 6):
        super(pix2pix_generator, self).__init__()

        down = []
        up = []

        self.downstack = []
        self.upstack = []

        temp_k = K

        # 3 -> 64
        down += [[nn.Conv2d(in_channels=input_channel,out_channels=temp_k,stride=2,kernel_size=4,padding=1),
                            nn.LeakyReLU(True)]]

        # 64 -> max 512
        for i in range(1,downsample_num):
            output_channel = min(512,temp_k*2)
            down += [network.cNsN_K(input_channels=temp_k,stride=2,N=4,k=output_channel,padding=1,norm=nn.BatchNorm2d, activation=nn.LeakyReLU(True))]
            temp_k = min(512,temp_k*2)


        for i in range(len(down)):
            setattr(self,'downstack'+str(i),nn.Sequential(*down[i]))
            self.downstack.append(getattr(self,'downstack'+str(i)))


        for i in range(1, downsample_num):
            if i ==1 and (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            elif (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            else:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=int(temp_k/2),stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(int(temp_k/2)),nn.ReLU(True)]]
                temp_k = int(temp_k / 2)

        for i in range(len(up)):
            setattr(self,'upstack'+str(i),nn.Sequential(*up[i]))
            self.upstack.append(getattr(self,'upstack'+str(i)))

        last = [nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=3,stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(3),nn.ReLU(True)]
        setattr(self,'last',nn.Sequential(*last))


    def forward(self,input):
        skips = []
        x = input

        for down in self.downstack:
            # pass x to each downsampling , then push it in to skips

            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        # reversed the skips, which mean the first one goes to the last one
        for up,skip in zip(self.upstack,skips):

            x = up(x)
            x = torch.cat((x,skip),1)

        x = self.last(x)
        return x




import net
import net.network as network
import net.resnet as resnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import copy
class global_frame_generator(nn.Module):
    #when bs = 1 BN = IN , we acutlly can use BN as default..
    def __init__(self,opt,input_channel,firstK=64,
                 n_downsample = 4,n_blocks = 9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        n_block is the num of the res-block
        n_downsample is the num of the downsample layer
        firstK is the first kernel num of the conv layer
        """
        assert(n_blocks>=0)
        super(global_frame_generator, self).__init__()
        # self.generate_first_frame = generate_first_frame
        ngf = firstK

        downsample = []
        downsample += network.c7s1_k(input_channel,ngf)
        # 7 * 7 output 64 CH
        # downsample layer
        temp_K = ngf

        for i in range(1,n_downsample+1):
            ds = network.dk(temp_K,temp_K*2)
            temp_K*=2
            downsample+=ds

        # downsample layer
        # res-block
        for i in range(1,n_blocks+1):
            res = [resnet.ResK(dim=temp_K,padding_type=padding_type,norm_layer=norm_layer)]
            downsample += res
        # res-block
        # upsample layer
        # should cat with the output from the encoder if the encoder using a vector Z
        self.downsample = nn.Sequential(*downsample)
        upsample = []
        for i in range(1,n_downsample+1):
            us = network.uk(input_channels=temp_K,stride=2,N=3,k=int(temp_K/2))
            temp_K = int(temp_K/2)
            upsample += us
        self.upsample = nn.Sequential(*upsample)
        # upsample layer

        last = network.c7s1_k(temp_K,opt.output_channel)
        self.last = nn.Sequential(*last)


    def forward(self,input,prev_frame,freature=None):
        """
        forward 1 big data will forward this many times.
        each "small" forward take the previous frames and (if use the last frames also) as the input
        pass though the init layer and the downsample layer
        then cat with the feature map from the Encoder (like a U-net but the feature are different)
        not sure should combine or not as quite like wasting caculation
        """
        x = input
        x = self.downsample(x)
        # inlcude resblock
        x = self.upsample(x)
        x = self.last(x)
        x = x + prev_frame
        return x



class pix2pix_generator(nn.Module):
    def __init__(self,input_channel ,K = 64 ,downsample_num = 6):
        super(pix2pix_generator, self).__init__()
        down = []
        up = []

        self.downstack = []
        self.upstack = []

        temp_k = K

        # 3 -> 64
        down += [[nn.Conv2d(in_channels=input_channel,out_channels=temp_k,stride=2,kernel_size=4,padding=1),
                            nn.LeakyReLU(True)]]

        # 64 -> max 512
        for i in range(1,downsample_num):
            output_channel = min(512,temp_k*2)
            down += [network.cNsN_K(input_channels=temp_k,stride=2,N=4,k=output_channel,padding=1,norm=nn.BatchNorm2d, activation=nn.LeakyReLU(True))]
            temp_k = min(512,temp_k*2)

        for i in range(len(down)):
            setattr(self,'downstack'+str(i),nn.Sequential(*down[i]))
            self.downstack.append(getattr(self,'downstack'+str(i)))


        for i in range(1, downsample_num):
            if i ==1 and (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            elif (2**(downsample_num-i)*(K/2)) >= 512:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=temp_k,stride=2,kernel_size=4,padding=1),nn.Dropout(0.5),nn.ReLU(True)]]
            else:
                up += [[nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=int(temp_k/2),stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(int(temp_k/2)),nn.ReLU(True)]]
                temp_k = int(temp_k / 2)

        for i in range(len(up)):
            setattr(self,'upstack'+str(i),nn.Sequential(*up[i]))
            self.upstack.append(getattr(self,'upstack'+str(i)))

        last = [nn.ConvTranspose2d(in_channels=temp_k*2,out_channels=3,stride=2,kernel_size=4,padding=1),nn.BatchNorm2d(3),nn.ReLU(True)]
        setattr(self,'last',nn.Sequential(*last))

    def forward(self,input):
        skips = []
        x = input

        for down in self.downstack:
            # pass x to each downsampling , then push it in to skips

            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        # reversed the skips, which mean the first one goes to the last one
        for up,skip in zip(self.upstack,skips):

            x = up(x)
            x = torch.cat((x,skip),1)

        x = self.last(x)
        return x
import net.SPADE as SPADE
class SPADE(model_wrapper):
    def __init__(self):
        super(SPADE, self).__init__()
    def initialize(self, opt):
        model_wrapper.initialize(self, opt)
        self.netG = SPADE.SPADEGenerator(opt).to(self.device)
        self.netG.init_weights(opt.init_type, opt.init_variance)
        if opt.use_vae:
            self.netE = SPADE.ConvEncoder(opt.input_chan).to(self.device)
            self.netE.init_weights(opt.init_type, opt.init_variance)
        if 'train' in self.opt.mode:
            self.netD = SPADE.MultiscaleDiscriminator(opt).to(self.device)
            self.netD.init_weights(opt.init_type, opt.init_variance)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            if opt.use_vae:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrain_path)
                self.KLDloss = net.loss.KLDLoss()
            # self.l1loss = nn.MSELoss()
            self.l1loss = nn.L1Loss()
            # try l2
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            self.GANloss = net.loss.GANLoss(device=self.device, lsgan=opt.lsgan)
            self.Tvloss = net.loss.TVLoss()
            G_params = list(self.netG.parameters()) + (list(self.netE.parameters()) if opt.use_vae else None)
            self.optimizer_G = torch.optim.Adam(G_params, lr=opt.learningrate, betas=(0.05, 0.999))
            self.optimizer_D = torch.optim.Adam(list(self.netD.parameters()), lr=opt.learningrate, betas=(0.05, 0.999))

        elif 'test' in self.opt.mode:
                self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
                if opt.use_vae:
                    self.load_network(self.netE, 'E', opt.which_epoch, self.save_dir)
        else:
            print('mode error,it would create a empty netG without pretrain params')

    def forward(self,input):
        label = input['label'].to(self.device) if self.opt.use_label else None  # CH = 3
        one_hot = input['one-hot'].to(self.device) if self.opt.use_label else None  # CH = label_CH
        edge = self.get_edges(label) if self.opt.use_label else None
        difference = input['difference'].to(self.device)
        current = input['current'].to(self.device)
        last = input['last'].to(self.device)
        next = input['next'].to(self.device)  # if the encoder need this?
        cat_feature = torch.cat([current, last, label, edge, one_hot], 1) if label is not None else torch.cat(
            [current, last], 1)
        cat_input = torch.cat([current, next, last],1)
        fake_next, KLD_loss = self.generate_fake(cat_input, cat_feature)
    def encode_z(self, target_sketch):
        mu, logvar = self.netE(target_sketch)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def generate_fake(self, cat_feature, cat_input):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(cat_input)
            KLD_loss = self.KLDloss(mu, logvar) * 0.05 # default lambda_kld = 0.05 in SPADE

        fake_image = self.netG(cat_feature, z=z)
        return fake_image, KLD_loss

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
        if self.opt.use_vae:
            self.save_network(self.netE, 'E', which_epoch, self.opt.gpu_ids)


    def update_learning_rate(self,epoch):
        lr = self.opt.learningrate * (0.1 ** (epoch // 10))
        for param_group in self.optimizer_D.param_groups:
         param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s learning rate is %s'%(epoch,lr))

# ******************** model ***************************** #
# this is only train with the difference
class pair_frame_generator(model_wrapper):
    def __init__(self):
        super(pair_frame_generator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def initialize(self,opt):
        model_wrapper.initialize(self, opt)
        G_input_chan = (opt.input_chan * 3) if opt.use_difference else opt.input_chan * 2
        self.netG = net.generator.pix2pix_generator(G_input_chan).to(self.device)
        self.netG.apply(init_weights)
        if 'train' in self.opt.mode:
            # should i cat with the difference?
            D_input_chan = opt.input_chan * 3 if opt.use_difference else opt.input_chan * 2
            #  (real,input) and the (fake,input)
            # input = cat(difference,current)
            # should i cat with the last frame?
            self.netD = net.discriminator.patchGAN(D_input_chan).to(self.device)
            self.netD.apply(init_weights)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.l1loss = nn.L1Loss()
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids) # default use L2loss
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
        current = input['current'].to(self.device)
        next = input['next'].to(self.device)
        last = input['last'].to(self.device)
        # difference = input['difference'].to(self.device)
        cat_input = torch.cat([current,last,difference],dim=1) if self.opt.use_difference else torch.cat([current,last],dim=1)
        fake_next = self.netG(cat_input)
        cat_fake = torch.cat([fake_next,difference,current],dim=1) if self.opt.use_difference else torch.cat([fake_next,current],dim=1)
        cat_real = torch.cat([next,difference,current],dim=1) if self.opt.use_difference else torch.cat([next,current],dim=1)
        dis_fake = self.netD(cat_fake.detach())
        dis_real = self.netD(cat_real)
        D_loss = (self.GANloss(dis_fake,False) + self.GANloss(dis_real,True)) * 0.5
        Vgg_loss = self.vggloss(fake_next,next)
        L1_loss = self.l1loss(fake_next,next) * 100.0
        new_dis_fake = self.netD(cat_fake)
        Gan_loss = self.GANloss(new_dis_fake,True)
        TV_loss = self.TVloss(fake_next)
        G_loss  = Gan_loss + L1_loss + Vgg_loss + TV_loss
        if self.opt.save_result:
            label = str(time.time())
            fast_check_result.imsave(current[-1,:,:,:], index=label + 'current', dir='./result/result_preview/')
            fast_check_result.imsave(fake_next[-1,:,:,:], index=label + 'fake', dir='./result/result_preview/')
            fast_check_result.imsave(next[-1,:,:,:], index=label + 'real', dir='./result/result_preview/')
            self.opt.save_result = False
        return {
            'G_loss':G_loss,
            'D_loss':D_loss,
            'Tv_loss':TV_loss,
            'L1_loss':L1_loss,
            'Gan_loss':Gan_loss,
            'Vgg_loss':Vgg_loss
        }
    def save(self,which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
    def update_learning_rate(self,epoch):
        lr = self.opt.learningrate * (0.1 ** (epoch // 10))
        for param_group in self.optimizer_D.param_groups:
         param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s learning rate is %s'%(epoch,lr))

# this is only train with the difference






class label_VAE(model_wrapper):
    def __init__(self):
        super(label_VAE, self).__init__()
    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        self.netG = net.generator.Deocder(opt).to(self.device)
        self.netG.apply(init_weights)
        self.ByteTensor = torch.cuda.ByteTensor if self.opt.gpu_ids > 0 else torch.ByteTensor
        if 'train' in self.opt.mode:
            self.netE = net.generator.ConvEncoder(opt).to(self.device)
            netD_inputchan = opt.input_chan * 3 + 2 if opt.use_label else opt.input_chan * 3
            # label + current + next/fake + label
            self.netD = net.discriminator.patchGAN(netD_inputchan).to(self.device)
            self.netD.apply(init_weights)
            netD_inputchan_seq = opt.input_chan * opt.n_past_frames
            # only cat the n * fake/real next frame is enough i guess?
            self.netD_seq = net.discriminator.patchGAN(netD_inputchan_seq).to(self.device)
            self.netD_seq.apply(init_weights)
            # need test a bit of this stuff
            # self.pool = torch.nn.MaxPool2d(3, stride=2, padding=[1, 1])
            self.pool = torch.nn.MaxPool2d(50, stride=2, padding=[1, 1])
            # need test a bit of this stuff
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.KLDloss = net.loss.KLDLoss()
            self.l1loss = nn.L1Loss()
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids)
            self.TVloss = net.loss.TVLoss()
            self.GANloss = net.loss.GANLoss(device=self.device, lsgan=opt.lsgan)
            params_G = list(self.netG.parameters()) + list(self.netE.parameters())
            params_D = list(self.netD.parameters()) + list(self.netD_seq.parameters())
            self.optimizer_G = torch.optim.Adam(params_G, lr=opt.learningrate, betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.Adam(params_D, lr=opt.learningrate, betas=(0.9, 0.999))
            print('---------- Networks initialized -------------')
            print('---------- NET G -------------')
            print(self.netG)
            print('---------- NET D -------------')
            print(self.netD)
            print('---------- NET E -------------')
            print(self.netE)
        elif 'test' in self.opt.mode :
            self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
        else:
            print('mode error,this sitaution will create a empty netG without any way pretrain params')
    def seq_optimize(self,input):
        # some init
        fake_frames = [] # this is all the fake_frames
        acc_loss = [] # this is all the pair loss(not include the seq_Dloss..)
        real_past_frames = [] # this is for the seq_D
        fake_past_empty_tensor = torch.zeros_like(input[0]['current']).to(self.device)
        # make a fake empty blank frames in the very beginning
        for i in range(self.opt.n_past_frames):
            fake_frames.append(fake_past_empty_tensor)
            real_past_frames.append(fake_past_empty_tensor)
        # some init

        # manyframesforward
        for j,each_frame in enumerate(input,start=0):
            # print(j) # count if the GPU memory will exceed... CRYING..
            real_past_frames[-1] = each_frame['next'].to(self.device) # every time the real_past_frames will auto update
            # generating fake
            if j == 0: # if is the first time we use the raw blank_frame/1st_frame/whatever given by the dataset
                loss, fake_next = self.pair_optimize(each_frame)
            else:
                # else the current will be replace by the fake_next_frame, which mean the previous fake_next_frame
                # is the next 'real'_current_frame
                each_frame['current'] = fake_next
                loss, fake_next = self.pair_optimize(each_frame)
            # generating fake
            # reload the list for compute the loss
            if j < self.opt.n_past_frames:
                fake_frames[j] = fake_next
                cat_fakes = torch.cat(fake_frames, 1)
            else:
                fake_frames.append(fake_next)
                cat_fakes = torch.cat(fake_frames[(-1 - self.opt.n_past_frames):-1], 1)
            # we take the past n-past frames from the list forever
            # reload the list for compute the loss
            # compute the D_seq_loss
            cat_reals = torch.cat(real_past_frames, 1)
            dis_fake = self.netD_seq(cat_fakes.detach())
            dis_real = self.netD_seq(cat_reals.detach())
            D_seq_loss = (self.GANloss(dis_fake, False) + self.GANloss(dis_real, True)) * 0.5
            dis_fake_seq_ = self.netD_seq(cat_fakes)
            GAN_Loss_seq = self.GANloss(dis_fake_seq_, True) * self.opt.GAN_lambda/self.opt.n_past_frames # should * 1/n-past-frame?
            # compute the D_seq_loss

            loss['D_Loss'] += D_seq_loss
            loss['G_Loss'] += GAN_Loss_seq
            acc_loss.append(loss)  # we will compute the loss at last
        # manyframesforward

        # combine the loss
        loss_dict = {'G_Loss':0,
            'D_Loss':0,
            'VGG_Loss':0,
            'L1_Loss':0
        }
        for _ in acc_loss: # all the pair loss are inside
            loss_dict['G_Loss'] += _['G_Loss']
            loss_dict['D_Loss'] += _['D_Loss']
            loss_dict['VGG_Loss'] += _['VGG_Loss']
            loss_dict['L1_Loss'] += _['L1_Loss']
        # combine the loss

        return loss_dict,fake_frames

    def pair_optimize(self,input):
        label = input['label'].to(self.device) if self.opt.use_label else None # CH = 3
        one_hot = input['one-hot'].to(self.device) if self.opt.use_label else None # CH = label_CH
        edge = self.get_edges(label) if self.opt.use_label else None
        difference = input['difference'].to(self.device)
        current = input['current'].to(self.device)
        last = input['last'].to(self.device)
        next = input['next'].to(self.device) # if the encoder need this?
        cat_feature = torch.cat([current,last,label,edge,one_hot],1) if label is not None else torch.cat([current,last],1)
        cat_input = torch.cat([current,next,last,difference,label,edge,one_hot],1) if label is not None else torch.cat([current,next,last,difference],1)
        # not sure if use label or one-hot as the input?
        fake_next, KLD_loss = self.generate_fake(cat_input,cat_feature)
        cat_fake = torch.cat([fake_next,current,last,label,edge],1) if self.opt.use_label else torch.cat([fake_next,current,last],1)
        cat_real = torch.cat([next,current,last,label,edge],1) if self.opt.use_label else torch.cat([next,current,last],1)
        dis_fake = self.netD(cat_fake.detach())
        dis_real = self.netD(cat_real.detach())
        # in the seq optimize the current is actually the fake current, so we detach it
        # and in pair optimize this effect nothing even it detach
        D_loss = (self.GANloss(dis_real,True) + self.GANloss(dis_fake,False)) * 0.5
        dis_fake_ = self.netD(cat_fake)
        GAN_Loss = self.GANloss(dis_fake_,True) * self.opt.GAN_lambda
        L1_Loss = self.l1loss(next,fake_next) * self.opt.l1_lambda
        TV_Loss = self.TVloss(fake_next)
        VGG_Loss = self.vggloss(fake_next,next) * self.opt.Vgg_lambda
        G_Loss = L1_Loss + VGG_Loss + GAN_Loss + KLD_loss + TV_Loss
        if self.opt.save_result:
            result_label = str(time.time())
            fast_check_result.imsave(next[-1,:,:,:],index=result_label+'real',dir='./result/result_preview/')
            fast_check_result.imsave(fake_next[-1,:,:,:],index=result_label+'fake',dir='./result/result_preview/')
            fast_check_result.imsave(current[-1,:,:,:],index=result_label+'current',dir='./result/result_preview/')
        return ({
            'G_Loss':G_Loss,
            'D_Loss':D_loss,
            'VGG_Loss':VGG_Loss,
            'L1_Loss':L1_Loss
        },fake_next) # this is for the seq
    def forward(self,input,mode='pair'):
        if mode == 'pair':
            return self.pair_optimize(input)
        elif mode == 'seq':
            return self.seq_optimize(input)

    def encode_z(self, x):
        mu, logvar = self.netE(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    def generate_fake(self,input,cat_feature):
        z, mu, logvar = self.encode_z(input)
        KLD_loss = self.KLDloss(mu, logvar)  # default lambda_kld = 0.05 in SPADE
        fake = self.netG(cat_feature,z)
        return fake, KLD_loss
    def update_learning_rate(self,epoch):
        lr = self.opt.learningrate * (0.1 ** (epoch // 10))
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('the current decay(not include the fixed rate epoch)_%s learning rate is %s' % (epoch, lr))
    def save(self,which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.opt.gpu_ids)
        self.save_network(self.netE, 'E', which_epoch, self.opt.gpu_ids)
    def get_edges(self,t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def test(self,input):
        with torch.no_grad():
            label = input['label'].to(self.device) if self.opt.use_label else None
            difference = input['difference'].to(self.device)
            current = input['current'].to(self.device)
            last = input['last'].to(self.device)
            next = input['next'].to(self.device)  # if the encoder need this?
            cat_feature = torch.cat([current, last, label], 1) if label is not None else torch.cat([current, last], 1)
            cat_input = torch.cat([current, next, last, difference, label], 1) if label is not None else torch.cat(
                [current, next, last, difference], 1)
            fake_next, KLD_loss = self.generate_fake(cat_input, cat_feature)
            return fake_next,current,next
    def inference(self,input,segmap_dir=None,mode='seq'):
        # we only need segmap(if use) and current+last
        # other stuff will randomly sampling from noraml distribution
        last = input['last'].to(self.device)
        if mode=='pair':
            current = input['current'].to(self.device)
            with torch.no_grad():
                if self.opt.use_label == False:
                    cat_feature = torch.cat([current, last], 1)
                    fake = self.netG(cat_feature,None)
                    label = None
                else:
                    assert segmap_dir is not None, 'if use_label please assign a segmap dir'
                    all_seg = [os.path.join(segmap_dir,seg) for seg in os.listdir(segmap_dir) if 'label' in seg and not 'segmap' in seg]
                    num = random.randint(1, 3)
                    label_list = []
                    for j in range(num):
                        index = random.randint(0, len(all_seg)-1)
                        label_list.append(all_seg.pop(index))
                    labels = [Image.open(img) for img in label_list]
                    pipes = []
                    pipes.append(transforms.Resize(self.opt.input_size))
                    pipes.append(transforms.ToTensor())
                    pipe = transforms.Compose(pipes)
                    all_label = [j[-1:, :, :].unsqueeze(0) for j in map(pipe, labels)]
                    slot = self.opt.label_CH - len(all_label)
                    empty = torch.zeros_like(all_label[0])
                    for i in range(slot):
                        all_label.append(empty)
                    one_hot = torch.cat(all_label,dim=1).to(self.device)
                    segmap = input['label'].to(self.device) if self.opt.use_label else None  # CH = 3
                    edge = self.get_edges(segmap)
                    cat_feature = torch.cat([current, last, one_hot,edge,segmap], 1)
                    # 6 + 10 + 1 + 1
                    fake = self.netG(cat_feature,None)

                return fake,one_hot
        else:
            first_frame = input['current'].to(self.device)
            fake_list = []
            if self.opt.use_label:
                assert segmap_dir is not None, 'if use_label please assign a segmap dir'
                for q in range(150): # this number should be modify in the future..
                    all_seg = [os.path.join(segmap_dir,seg) for seg in os.listdir(segmap_dir) if 'label' in seg and not 'segmap' in seg]
                    num = random.randint(1, 3)
                    label_list = []
                    for j in range(num):
                        index = random.randint(0, len(all_seg)-1)
                        label_list.append(all_seg.pop(index))
                    labels = [Image.open(img) for img in label_list]
                    pipes = []
                    pipes.append(transforms.Resize(self.opt.input_size))
                    pipes.append(transforms.ToTensor())
                    pipe = transforms.Compose(pipes)
                    all_label = [j[-1:, :, :].unsqueeze(0) for j in map(pipe, labels)]
                    slot = self.opt.label_CH - len(all_label)
                    empty = torch.zeros_like(all_label[0])
                    for i in range(slot):
                        all_label.append(empty)
                    one_hot = torch.cat(all_label,dim=1).to(self.device)
                    segmap = input['label'].to(self.device)
                    edge = self.get_edges(segmap)
                    if q == 0:
                        cat_feature = torch.cat([first_frame, last, one_hot,edge,segmap], 1)
                        fake = self.netG(cat_feature,None)
                    else:
                        cat_feature = torch.cat([fake_list[-1],last,one_hot,edge,segmap],1)
                        fake = self.netG(cat_feature,None)
                    fake_list.append(fake)

            return fake_list
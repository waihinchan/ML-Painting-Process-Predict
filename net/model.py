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
import matplotlib
import matplotlib.pyplot as plt
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

# color to sketch
class colortosketch(model_wrapper):
    def __init__(self):
        super(colortosketch, self).__init__()
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
            self.vggloss = net.loss.pixHDversion_perceptual_loss(opt.gpu_ids,loss=nn.L1Loss())
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
        dis_fake_ = self.netD(cat_fake)
        gan_loss = self.GANloss(dis_fake_,True)
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
        print('the current decay(not include the fixed rate epoch)_%s learning rate is %s'%(epoch,lr))
    def inference(self,input):
        if isinstance(input,str):
            import os
            assert os.path.isfile(input)
            input_image = fast_check_result.grabdata(self.opt,input)
        elif isinstance(input,torch.Tensor):
            input_image = input
        else:
            print('please pass a image path or a tensor')
            return None

        input_image = input_image.to(self.device)
        with torch.no_grad():
            generated = self.netG(input_image)

        return generated
# color to sketch

# ********************* main model ******************************************
class SCAR(model_wrapper):
    def __init__(self):
        super(SCAR, self).__init__()
    def initialize(self,opt):
        model_wrapper.initialize(self,opt)
        self.netG = net.generator.Decoder2(opt).to(self.device)
        self.netG.apply(init_weights)
        # # just a small test
        # self.netG = net.generator.Deocder3(opt,14).to(self.device)
        # self.netG.apply(init_weights)
        # # just a small test
        self.ByteTensor = torch.cuda.ByteTensor if self.opt.gpu_ids > 0 else torch.ByteTensor
        if 'train' in self.opt.mode:
            self.netE = net.generator.Encoder2(opt).to(self.device)
            self.netE.apply(init_weights) # if use encoder3 here should comment
            netD_inputchan = opt.input_chan * 3 # current last real_next/fake_next 9
            if opt.use_difference:
                netD_inputchan += opt.input_chan
            if opt.use_label:
                netD_inputchan += 2 # instance + segmap 11
            if opt.use_wireframe:
                netD_inputchan += 1 # 12
            if opt.use_degree =='wrt_position':
                assert opt.use_difference,'if use degree,please use difference'
                netD_inputchan += (opt.granularity+1) # 17

            # the cat input will be current + next + last + full_label_map + edge + degree + labelCH(if degree is None or time)
            self.netD = net.discriminator.patchGAN(netD_inputchan).to(self.device)
            self.netD.apply(init_weights)
            netD_inputchan_seq = opt.output_channel * opt.n_past_frames
            self.netD_seq = net.discriminator.patchGAN(netD_inputchan_seq).to(self.device)
            self.netD_seq.apply(init_weights)
            pretrain_path = '' if self.opt.mode == 'train' else self.save_dir
            self.load_network(self.netD, 'D', opt.which_epoch, pretrain_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrain_path)
            self.load_network(self.netE, 'E', opt.which_epoch, pretrain_path)
            self.load_network(self.netD_seq, 'D_seq', opt.which_epoch, pretrain_path)
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
        elif 'test' in self.opt.mode:
            self.load_network(self.netG, 'G', opt.which_epoch, self.save_dir)
        else:
            print('mode error,this sitaution will create a empty netG without any way pretrain params')
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
        self.save_network(self.netD_seq, 'D_seq', which_epoch, self.opt.gpu_ids)
# *******************************some init function ******************************************* #

    def forward(self,input,mode='pair'):
        if mode == 'pair':
            return self.pair_optimize(input)
        elif mode == 'seq':
            return self.seq_optimize(input)
    def pre_process_input(self,input):
        current = input['current'].to(self.device)
        last = input['last'].to(self.device)
        input_list = [current,last] # the next remain to modify at below
        next = input['next'].to(self.device)
        # ****************** current next last are fixed ******************

        if self.opt.use_difference:
            difference = input['difference'].to(self.device) # the difference remain to modify at below
        else :
            difference = None
        if self.opt.use_label: # this is the full segmap,not the one-hot
            label = input['label'].to(self.device)
            input_list.append(label)
        else:
            label = None
        if self.opt.use_degree == 'wrt_position':
            assert input['segmaps'] is not None, 'if use wrt_position degree please return a single segmap list'
            segmaps = [segmap.to(self.device) for segmap in input['segmaps']]
            degree = self.caculate_degree(difference,segmaps)
            input_list.append(degree)
        else:
            degree = None
        if self.opt.use_instance:
            assert label is not None,'if use instance please return a full_segmap'
            # TODO actually the instance map is not totaly the same as full segmap. but in my case is the same
            instance = self.get_edges(label)
            input_list.append(instance)
        else:
            instance = None
        if self.opt.use_wireframe:
            wire_frame = input['use_wireframe']
            input_list.append(wire_frame)
        else:
            wire_frame = None

        E_list = input_list +[next] if difference is None else input_list+[next,difference]
        cat_list = input_list
        Encoder_input = torch.cat(E_list,dim=1)
        Decoder_input = torch.cat(input_list,dim=1)

        return {
            'current':current, # tensor
            'last':last, # tensor 
            'next':next, # tensor 
            'label':label, # tensor 
            'difference':difference, # tensor 
            'instance': instance, # tensor 
            'degree':degree, # tensor
            'wire_frame':wire_frame, # tensor
            'Encoder_input': Encoder_input, #tensor
            'Decoder_input':Decoder_input, # tensor
            'cat_list':cat_list # list
        }
    def get_edges(self,t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()
    def generate_next_frame(self,E_input,G_input):
        fake_next,weight,KLD_Loss = self.generate_fake(E_input,G_input)
        return fake_next,weight,KLD_Loss
    def seq_optimize(self,input):
        # ************************** some init ************************** #
        fake_frames = []  # this is all the fake_frames
        acc_loss = []  # this is all the pair loss(not include the seq_Dloss..)
        real_past_frames = []  # this is for the seq_D
        fake_past_empty_tensor = torch.zeros_like(input[0]['current']).to(self.device) # make a fake empty blank frames in the very beginning
        for i in range(self.opt.n_past_frames):
            fake_frames.append(fake_past_empty_tensor)
            real_past_frames.append(fake_past_empty_tensor)
        # ************************** some init ************************** #

        # ************************** many frames forward ************************** #
        for j, each_frame in enumerate(input, start=0):
            # print(j) # count if the GPU memory will exceed... CRYING..
            real_past_frames[-1] = each_frame['next'].to(self.device)  # every time the real_past_frames will auto update
            if j == 0:  # if is the first time we use the raw blank_frame/1st_frame/whatever given by the dataset
                loss, fake_next = self.pair_optimize(each_frame)
            else:
                each_frame['current'] = fake_next
                # else the current will be replace by the fake_next_frame, which mean the previous fake_next_frame
                # is the next 'real'_current_frame
                loss, fake_next = self.pair_optimize(each_frame)
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
            GAN_Loss_seq = self.GANloss(dis_fake_seq_,True) * self.opt.GAN_lambda / self.opt.n_past_frames  # should * 1/n-past-frame?
            # compute the D_seq_loss
            loss['D_loss'] += D_seq_loss
            loss['G_loss'] += GAN_Loss_seq
            acc_loss.append(loss)  # we will compute the loss at last
        # combine the loss
        loss_dict = {'G_loss': 0,
                     'D_loss': 0
                    
                     }
        for _ in acc_loss:  # all the pair loss are inside
            loss_dict['G_loss'] += _['G_loss']
            loss_dict['D_loss'] += _['D_loss']
        # combine the loss
        if self.opt.save_result:
          result_label = str(time.time())
          folder = './result/result_preview/' + result_label
          os.mkdir(folder)
          for k,save in enumerate(fake_frames,start=0):
            fast_check_result.imsave(save[-1,:,:,:],index=str(k),dir=folder+'/')

        return loss_dict,fake_frames

    def pair_optimize(self, input):
        input_ = self.pre_process_input(input)
        # generate fake
        fake,weight,KLD_Loss = self.generate_next_frame(input_['Encoder_input'],input_['Decoder_input'])
        # pass to the D
        if not self.opt.use_raw_only:
            fake_next = fake*weight + (1-weight) * input_['current']
        else:
            fake_next = fake
        raw_cat_real = input_['cat_list']
        raw_cat_fake = input_['cat_list']
        # replace the fake / real next
        raw_cat_real += [input_['next']]
        raw_cat_fake += [fake_next]
        cat_fake = torch.cat(raw_cat_fake,1)
        cat_real = torch.cat(raw_cat_real,1)
        dis_fake = self.netD(cat_fake.detach())
        dis_real = self.netD(cat_real.detach())
        # caculate the loss
        D_loss = (self.GANloss(dis_real,True) + self.GANloss(dis_fake,False)) * 0.5
        dis_fake_ = self.netD(cat_fake)
        GAN_Loss = self.GANloss(dis_fake_,True) * self.opt.GAN_lambda
        L1_Loss = self.l1loss(input_['next'],fake_next) * self.opt.l1_lambda
        TV_Loss = self.TVloss(fake_next)
        VGG_Loss = self.vggloss(fake_next,input_['next']) * self.opt.Vgg_lambda
        G_Loss = L1_Loss + VGG_Loss + GAN_Loss + KLD_Loss + TV_Loss
        if self.opt.save_result:
            result_label = str(time.time())
            fast_check_result.imsave(input_['next'][-1,:,:,:],index=result_label+'real',dir='./result/result_preview/')
            fast_check_result.imsave(fake_next[-1,:,:,:],index=result_label+'fake',dir='./result/result_preview/')
            fast_check_result.imsave(input_['current'][-1,:,:,:],index=result_label+'current',dir='./result/result_preview/')
            # self.opt.save_result = False
            # if self.opt.use_difference:
            #     fast_check_result.imsave(input_['difference'][-1,:,:,:],index=result_label+'difference',dir='./result/result_preview/')
            # if self.opt.use_degree == 'wrt_position':
            #     for k in range(self.opt.granularity + 1):
            #         fast_check_result.imsave(input_['degree'][-1, k, :, :], index=result_label + 'degree' + str(k),dir='./result/result_preview/')

        return {
            'G_loss':G_Loss,
            'D_loss':D_loss,
             'KLd_loss':KLD_Loss
        },fake_next
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
        KLD_loss = self.KLDloss(mu, logvar) * self.opt.Kld_lambda  # default lambda_kld = 0.05 in SPADE
        fake,w = self.netG(cat_feature,z)
        return fake, w, KLD_loss
    def caculate_degree(self,difference,segmap_list):
        assert self.opt.use_degree == 'wrt_position','if not wrt_position please return the degree'
        """
        # step1: return the no zero index of difference
        # step2: caculate the percentage of the each contribution to the one-hot label
        # step3: according to the granularity to cacualte the degree
        # return a multi CH(granularity) tensor
        """
        # TODO should make a one-hot base on the full-segmap not single of them..
        #  but don't know why SPADE's pre-process not working
        difference_ = torch.sum(difference,1,keepdim=True) # merge the difference CH
        empty = torch.zeros(self.opt.batchSize, 1, self.opt.input_size, self.opt.input_size).to(self.device) # if no tensor at a degree, fill a zero
        degree_list = [self.opt.zero_degree] # the least degree0 is less than 1%
        one_hot_list = [[empty]] #the first empty one
        for i in range(1,self.opt.granularity+1): # we already have zero-0.01
            degree_list.append((0.99/self.opt.granularity)*i)
            one_hot_list.append([empty]) # this is for later we merge the all single_segmap in 1 degree list

        for single_segmap in segmap_list:
            field = single_segmap[difference_ != 0]
            percentage = len(field.nonzero())/len(single_segmap.nonzero())
            for degree,_ in enumerate(degree_list,start=0):
                if percentage <= _:
                    one_hot_list[degree].append(single_segmap) # degreeN is list, append that tensor
                    break
        cat_list = []
        for each_degree in one_hot_list:
            merge = torch.cat(each_degree,dim=1)
            _ = torch.sum(merge,1,keepdim=True)
            cat_list += [_]
        return torch.cat(cat_list,dim=1) # the shape should be match to granularity + 1(one is the zeros)
    def make_random_degree(self,segmap_list):
      """
      we don't need difference, just make randomly put the segmap in different degree
      """
      # self.opt.granularity+1
      segmaps = [segmap.to(self.device) for segmap in segmap_list]

      empty = torch.zeros(self.opt.batchSize, 1, self.opt.input_size, self.opt.input_size).to(self.device) # if no tensor at a degree, fill a zero
      one_hot_list = []
      for i in range(self.opt.granularity+1):
        one_hot_list.append([empty]) # fill all the slot first
      # random number random index random degree
      import random
      while len(segmaps)!=0:
        num = random.randint(0,len(segmaps)) # how many segmap will go to the below degree
        degree = random.randint(0,self.opt.granularity)
        for j in range(num):
          index = random.randint(0,len(segmaps)-1) # because we used pop, so the length will change
          one_hot_list[degree].append(segmaps.pop(index))
      cat_list = []
      for each_degree in one_hot_list:
        _ = torch.cat(each_degree,dim=1)
        merge = torch.sum(_,1,keepdim=True) # we merge all the segmap in one degree
        cat_list += [merge]
      return torch.cat(cat_list,dim=1)
    def pre_process_input_(self,input):
        current = input['current'].to(self.device)
        last = input['last'].to(self.device)
        input_list = [current,last] # the next remain to modify at below
        next = input['next'].to(self.device)
        # ****************** current next last are fixed ******************

        if self.opt.use_difference:
            difference = input['difference'].to(self.device) # the difference remain to modify at below
        else :
            difference = None
        if self.opt.use_label: # this is the full segmap,not the one-hot
            label = input['label'].to(self.device)
            input_list.append(label)
        else:
            label = None
        if self.opt.use_degree == 'wrt_position':
            assert input['segmaps'] is not None, 'if use wrt_position degree please return a single segmap list'
            segmaps = [segmap.to(self.device) for segmap in input['segmaps']]
            degree = self.make_random_degree(segmaps) #########here is the change
            input_list.append(degree)
        else:
            degree = None
        if self.opt.use_instance:
            assert label is not None,'if use instance please return a full_segmap'
            # TODO actually the instance map is not totaly the same as full segmap. but in my case is the same
            instance = self.get_edges(label)
            input_list.append(instance)
        else:
            instance = None
        if self.opt.use_wireframe:
            wire_frame = input['use_wireframe']
            input_list.append(wire_frame)
        else:
            wire_frame = None

        E_list = input_list +[next] if difference is None else input_list+[next,difference]
        cat_list = input_list
        Encoder_input = torch.cat(E_list,dim=1)
        Decoder_input = torch.cat(input_list,dim=1)

        return {
            'current':current,
            'last':last,
            'next':next,
            'label':label,
            'difference':difference,
            'instance': instance,
            'degree':degree,
            'wire_frame':wire_frame,
            'Encoder_input': Encoder_input,
            'Decoder_input':Decoder_input,
            'cat_list':cat_list
        }
    def test(self,input):
      # self.netE = net.generator.Encoder2(self.opt).to(self.device)
      # pretrain_path = self.save_dir
      # self.load_network(self.netE, 'E', self.opt.which_epoch, pretrain_path)
      # self.KLDloss = net.loss.KLDLoss()
      with torch.no_grad():
        input_ = self.pre_process_input_(input)
        # fake,weight,KLD_Loss = self.generate_next_frame(input_['Encoder_input'],input_['Decoder_input'])
        fake,weight = self.netG(input_['Decoder_input'],None)
        print(input_['Decoder_input'].shape)
        if not self.opt.use_raw_only:
            fake_next = fake*weight + (1-weight) * input_['current']
        else:
            fake_next = fake

      return fake_next

    def inference(self,input,segmap_list,times=30):
      with torch.no_grad():
        fake_frames = []
        current = input['current'].to(self.device)
        last = input['last'].to(self.device)
        fixed_input = [last]
        if self.opt.use_label:
          label = input['label'].to(self.device)
          fixed_input.append(label)
          if self.opt.use_instance:
            instance = self.get_edges(label).to(self.device)
            fixed_input.append(instance)
        if self.opt.use_wireframe:
          wire_frame = input['wire_frame'].to(self.device)
          fixed_input.append(instance)
      # current and degree will refresh every time, the last and segmap and wireframe is fixed

        for i in range(times):
          random_degree = None # just make sure everytime is the new one.. actually should't need it
          dynamic_input = None
          random_degree = self.make_random_degree(segmap_list) if self.opt.use_degree else None
          dynamic_input = fixed_input + [current,random_degree] if self.opt.use_degree else fixed_input + [current]
          cat_feature = torch.cat(dynamic_input,dim=1)
          fake,weight = self.netG(cat_feature,None)
          if not self.opt.use_raw_only:
              fake_next = fake*weight + (1-weight) * current
          else:
              fake_next = fake
          fake_frames.append(fake_next)
          # imsave(fake_next[-1,:,:,:],index = 'fake'+str(i),dir = './result/result_preview/')
          # imsave(current[-1,:,:,:],index = 'current'+str(i),dir = './result/result_preview/')
          # imsave(label[-1,:,:,:],index = 'label'+str(i),dir = './result/result_preview/')
          # imsave(last[-1,:,:,:],index = 'last'+str(i),dir = './result/result_preview/')

          # for j in range(random_degree.size(1)):
          #   imsave(random_degree[-1,j,:,:],index = 'degree'+str(i) + '_' + str(j) , dir = './result/result_preview/')

          current = fake_next
          
        return fake_frames

from utils.fast_check_result import imsave
# this is for inference mode... should update in the future
import os
import random
from torchvision import transforms
from PIL import Image
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
# this is for inference mode...



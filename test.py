from mydataprocess import mydataloader
import option
import net.model
import os
import time
from PIL import Image
from torchvision import transforms
import torch
unloader = transforms.ToPILImage()
def imsave(tensor,index):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save('./result/video_result/' + str(index) + '.jpg')

myoption = option.opt()
myoption.mode = 'test'
myoption.which_epoch = '160'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

mymodel = net.model.SCAR()
mymodel.initialize(opt = myoption)

# test data
pipes = []
pipes.append(transforms.Resize(myoption.input_size))
pipes.append(transforms.ToTensor())
pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5)))
pipe = transforms.Compose(pipes)

test_tensor = pipe(Image.open("./dataset/step/_109/8.jpg")).unsqueeze(0).to(mymodel.device)
# test data

generator = mymodel.netG
target = test_tensor
forward_fake_frames = []
for j in range(0, myoption.n_past_frames):
    forward_fake_frames += [torch.zeros_like(target).to(mymodel.device)]

    # blank image at first
forward_fake_frames += [target]
fake_frames = []
for i in range(0,myoption.bs_total_frames):
        input = None
        input = torch.cat(forward_fake_frames, dim=1).detach()
        G_output = None
        prev_frame = forward_fake_frames[-1] if i == 0 else fake_frames[-1]
        G_output = mymodel.netG(input,prev_frame)

        forward_fake_frames[-1] = G_output
        fake_frames.append(G_output)
        imsave(G_output,i)




















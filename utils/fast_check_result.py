from PIL import Image
from torchvision import transforms
import torch
unloader = transforms.ToPILImage()
def imsave(tensor,index,dir = './result/video_result/'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = torch.clamp(image,0,1)
    image = unloader(image)
    image.save( dir+ str(index) + '.jpg')

def grabdata(opt,path):
    input_image = Image.open(path)
    pipe = []
    pipe.append(transforms.Resize(opt.input_size))
    pipe.append(transforms.ToTensor())

    # pipe.append(transforms.Normalize((0.5, 0.5, 0.5),
    #                                   (0.5, 0.5, 0.5)))
    pipe = transforms.Compose(pipe)
    image = pipe(input_image)

    return image.unsqueeze(0)
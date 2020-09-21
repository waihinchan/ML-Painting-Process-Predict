from PIL import Image
from torchvision import transforms
unloader = transforms.ToPILImage()
def imsave(tensor,index,dir = './result/video_result/'):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save( dir+ str(index) + '.jpg')

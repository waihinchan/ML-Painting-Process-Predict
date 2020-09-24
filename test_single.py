import option
import net.model
from PIL import Image
from torchvision import transforms
unloader = transforms.ToPILImage()
def imsave(tensor,index):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save('./result/single/' + str(index) + '.jpg')

myoption = option.opt()
myoption.name = 'single_frame'
myoption.mode = 'test'
myoption.which_epoch = '190'
for name,value in vars(myoption).items():
    print('%s=%s' % (name,value))

mymodel = net.model.single_frame()
mymodel.initialize(opt = myoption)

# test data
pipes = []
pipes.append(transforms.Resize((512,512)))
pipes.append(transforms.ToTensor())
# pipes.append(transforms.Normalize((0.5, 0.5, 0.5),
#                                   (0.5, 0.5, 0.5)))
pipe = transforms.Compose(pipes)

test_tensor = pipe(Image.open('/home/waihinchan/Desktop/123.jpeg')).unsqueeze(0).to(mymodel.device)
# test data
print(test_tensor.shape)
generator = mymodel.netG

G_out = generator(test_tensor)
imsave(G_out,'single_test')




















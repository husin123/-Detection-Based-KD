import torch,os
from torchvision import transforms
from PIL import ImageDraw
def detection_vis(tensor, box, label, output_path, name):
    assert tensor.ndim == 3 and tensor.shape[0] == 3

    mean = [123.675/255, 116.28/255, 103.53/255]
    std = [58.395/255, 57.12/255, 57.375/255]
    tensor = tensor.clone().detach().cpu()
    tensor.mul_(torch.Tensor(std)[:, None, None]).add_(
        torch.Tensor(mean)[:, None, None]
    )
    torch.clip_(tensor, 0, 1)
    changer = transforms.ToPILImage()
    img = changer(tensor)
    a = ImageDraw.ImageDraw(img)
    for l,b in zip(label,box):
        min_x, min_y, max_x, max_y = torch.split(
            b, 1, dim=-1)
        min_x,min_y,max_x,max_y = min_x.item() ,min_y.item() ,max_x.item() ,max_y.item()
        a.rectangle((min_x,min_y,max_x,max_y),outline="red",width=5)
        a.text((min_x/2+max_x/2,min_y/2+max_y/2),str(l.item()))
    img.save(os.path.join(output_path, name) + ".png")
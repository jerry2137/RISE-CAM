import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import SmoothGradCAMpp

cudnn.benchmark = True

# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)

# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

# Returns normalized Area Under Curve of the first 20 Percent of the array
def auc(arr, fraction=1.0):
    """Returns normalized Area Under Curve of the top fraction of the array."""
    arr = arr[:int(len(arr) * (fraction))]
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

# Function that blurs input image
def blur(x, klen=11, nsig=5):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    kern = torch.from_numpy(kern.astype('float32'))
    return nn.functional.conv2d(x, kern, padding=klen//2)

# Insertion and deletion
class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, hw, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.cuda())
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (hw + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, hw), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start.cuda())
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, hw)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, hw)[0, :, coords]
        return scores

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath, p1=0.1):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = p1

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()

        # Apply the masks on the image.
        stack = torch.mul(self.masks, x.data)

        # Feed the masked images into the model.
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        CL = p.size(1) # Number of classes

        # Multiply the confidence of a certain class with the masks and sum them up to get the saliency maps.
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal

class RISECAM(RISE):
    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        masked_imgs = torch.mul(self.masks, x.data)

        masked_imgs.requires_grad = True

        scores = []
        with SmoothGradCAMpp(self.model) as cam_extractor:
            _, class_number = torch.topk(self.model(x), k=1)
            class_number = class_number[0]

            for masked_img in tqdm(masked_imgs):
                scores.append(cam_extractor(class_number[0].item(), self.model(masked_img.unsqueeze(0)))[0][0])
            scores = torch.stack(scores)

        _, feature_height, feature_width = scores.size()
        sals = torch.matmul(
            scores.view(N, feature_height * feature_width).transpose(0, 1),
            self.masks.view(N, H * W)
        )
        sals = sals.view(feature_height, feature_width, H, W)
        sals = sals / N / self.p1
        return sals



def rise(model, input_tensor, input_size=(224, 224), N=6000, s=8, p1=0.1, mask_path=''):
    # Freeze the gradient.
    for p in model.parameters():
        p.requires_grad = False

    # Create explainer
    rise_explainer = RISE(model, input_size)

    # Generate/load masks for RISE.
    if(os.path.isfile(mask_path) & mask_path.endswith('.npy')):
        print('Loading masks.')
        rise_explainer.load_masks(mask_path)
    else:
        print('Mask path incorrect (not .npy) or not given, generating new masks.')
        rise_explainer.generate_masks(N=N, s=s, p1=p1, savepath='masks_{}.npy'.format(N))
    
    # Get the predicted class.
    _, class_number = torch.topk(model(input_tensor.cuda()), k=1)
    class_number = class_number[0]

    # Generate saliency map.
    saliency_maps = rise_explainer(input_tensor.cuda()).cpu().numpy()
    saliency_map = saliency_maps[class_number[0]]
    return saliency_map


def gradcam(model, input_tensor, input_size=(224, 224)):
    # Freeze the gradient.
    for p in model.parameters():
        p.requires_grad = False

    # Generate saliency map.
    input_tensor.requires_grad = True
    with SmoothGradCAMpp(model) as cam_extractor:
        out = model(input_tensor.cuda())
        _, class_number = torch.topk(out, k=1)
        class_number = class_number[0]
        sal = cam_extractor(class_number[0].item(), out)
    input_tensor.requires_grad = False

    overlay = to_pil_image(sal[0][0], mode='F').resize(input_size, resample=Image.BICUBIC)
    saliency_map = np.asarray(overlay)
    return saliency_map


def risecam(model, input_tensor, top_k='auto', input_size=(224, 224), N=6000, s=8, p1=0.1, mask_path=''):
    # Freeze the gradient.
    for p in model.parameters():
        p.requires_grad = False
    
    # Create explainer.
    risecam_explainer = RISECAM(model, input_size)
    
    # Generate/load masks for RiseCAM.
    if(os.path.isfile(mask_path) & mask_path.endswith('.npy')):
        risecam_explainer.load_masks(mask_path)
    else:
        print('Mask path incorrect (not .npy) or not given, generating new masks.')
        risecam_explainer.generate_masks(N=N, s=s, p1=p1, savepath='masks_{}.npy'.format(N))

    # Generate saliency map for the features.
    input_tensor.requires_grad = True
    feature_saliency_maps = risecam_explainer(input_tensor.cuda()).cpu().numpy()

    # Generate GradCAM saliency map.
    with SmoothGradCAMpp(model) as cam_extractor:
        out = model(input_tensor.cuda())
        _, class_number = torch.topk(out, k=1)
        class_number = class_number[0]
        gradcam_sal = np.mean([cam_extractor(class_number[0].item(), out)[0][0].cpu().numpy() for _ in range(100)], axis=0)
    input_tensor.requires_grad = False
    
    # Flatten the saliency maps.
    feature_height, feature_width, height, width = feature_saliency_maps.shape
    gradcam_sal_flatten = gradcam_sal.reshape(feature_height*feature_width)
    feature_saliency_flatten = feature_saliency_maps.reshape(feature_height*feature_width, height, width)

    # Return all the saliency maps sorted by the GradCAM saliency map.
    if(top_k == 'all'):
        indices = np.argsort(gradcam_sal_flatten)[::-1]
        saliency_maps_sorted = feature_saliency_flatten[indices]
        return saliency_maps_sorted
    
    # Auto select top k value and return the sum of the top k feature saliency maps.
    if(top_k == 'auto'):
        indices = np.argsort(gradcam_sal_flatten)[::-1]
        saliency_maps_sorted = feature_saliency_flatten[indices]

        # Find the k value that has the highest score.
        saliency_map_sum = np.zeros(saliency_maps_sorted[0].shape)
        best_score = -np.Inf
        best_k = 0
        for i, saliency_map in enumerate(saliency_maps_sorted):
            saliency_map_sum += saliency_map
            insertion = CausalMetric(model, 'ins', height, substrate_fn=blur)
            deletion = CausalMetric(model, 'del', height, substrate_fn=torch.zeros_like)
            insertion_result = insertion.single_run(input_tensor, saliency_map_sum, height*width)
            deletion_result = deletion.single_run(input_tensor, saliency_map_sum, height*width)
            score = auc(insertion_result, fraction=0.2) - auc(deletion_result, fraction=0.2)
            if(score > best_score):
                best_score = score
                best_k = i + 1
                saliency_map_best = saliency_map_sum.copy()
        print('best value for k is: {}'.format(best_k))

        return saliency_map_best
    
        
    # Auto select top k value within top 20% and return the sum of the top k feature saliency maps.
    if(top_k == 'optimal'):
        indices = np.argsort(gradcam_sal_flatten)[::-1]
        saliency_maps_sorted = feature_saliency_flatten[indices]

        # Selct the top 20% only.
        fraction = 0.2
        saliency_maps_sorted = saliency_maps_sorted[:int(len(saliency_maps_sorted) * (fraction))]

        # Find the k value that has the highest score.
        saliency_map_sum = np.zeros(saliency_maps_sorted[0].shape)
        best_score = -np.Inf
        best_k = 0
        for i, saliency_map in enumerate(saliency_maps_sorted):
            saliency_map_sum += saliency_map
            insertion = CausalMetric(model, 'ins', height, substrate_fn=blur)
            deletion = CausalMetric(model, 'del', height, substrate_fn=torch.zeros_like)
            insertion_result = insertion.single_run(input_tensor, saliency_map_sum, height*width)
            deletion_result = deletion.single_run(input_tensor, saliency_map_sum, height*width)
            score = auc(insertion_result, fraction=0.2) - auc(deletion_result, fraction=0.2)
            if(score > best_score):
                best_score = score
                best_k = i + 1
                saliency_map_best = saliency_map_sum.copy()
        print('best value for k is: {}'.format(best_k))

        return saliency_map_best
    
    # Return the sum of the top k feature saliency maps.
    indices = np.argpartition(gradcam_sal_flatten,-top_k)[-top_k:]
    saliency_map = np.mean(feature_saliency_flatten[indices], axis=0)
    return saliency_map
import os
import copy
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

from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import SmoothGradCAMpp

cudnn.benchmark = True

# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    lambda x: x.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

# Load black box model for explanations
def get_model():
    model = models.resnet50(True)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

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

# Returns normalized Area Under Curve of the top fraction of the array
def auc(arr, fraction=1.0):
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
    def __init__(self, input_size, gpu_batch=100):
        super(RISE, self).__init__()
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
        model = get_model()
        for i in range(0, N, self.gpu_batch):
            p.append(model(stack[i:min(i + self.gpu_batch, N)]))
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
        self.model = get_model()
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


def rise(input_tensor, input_size=(224, 224), N=6000, s=8, p1=0.1, mask_path=''):

    # Create explainer
    rise_explainer = RISE(input_size)

    # Generate/load masks for RISE.
    if(os.path.isfile(mask_path) & mask_path.endswith('.npy')):
        print('Loading masks.')
        rise_explainer.load_masks(mask_path)
    else:
        print('Mask path incorrect (not .npy) or not given, generating new masks.')
        rise_explainer.generate_masks(N=N, s=s, p1=p1, savepath='masks_{}.npy'.format(N))
    
    # Get the predicted class.
    model = get_model()
    _, class_number = torch.topk(model(input_tensor.cuda()), k=1)
    class_number = class_number[0]

    # Generate saliency map.
    saliency_maps = rise_explainer(input_tensor.cuda()).cpu().numpy()
    saliency_map = saliency_maps[class_number[0]]
    return saliency_map


def gradcam(input_tensor, input_size=(224, 224), new_risecam=False):
    model = get_model()
    # Generate saliency map.
    input_tensor.requires_grad = True
    with SmoothGradCAMpp(model) as cam_extractor:
        out = model(input_tensor.cuda())
        _, class_number = torch.topk(out, k=1)
        class_number = class_number[0]
        sal = cam_extractor(class_number[0].item(), out)
        layer_name = cam_extractor.target_names[0]
    input_tensor.requires_grad = False

    if new_risecam:
        gradcam_sal_flatten = sal[0].flatten()
        return layer_name, gradcam_sal_flatten.cpu()

    overlay = to_pil_image(sal[0][0], mode='F')
    overlay = overlay.resize(input_size, resample=Image.BICUBIC)
    return  np.asarray(overlay)


def risecam(input_tensor, top_k='optimal', input_size=(224, 224), N=6000, s=8, p1=0.1, mask_path=''):
    
    # Create explainer.
    risecam_explainer = RISECAM(input_size)
    
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
    model = get_model()
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
            model = get_model()
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


def get_hidden_features(x, layer, model):
    activation = {}

    def get_activation(name):
        def hook(m, i, o):
            activation[name] = o.detach()
        return hook

    model.get_submodule(layer).register_forward_hook(get_activation(layer))
    _ = model(x)
    return activation[layer]

def generate_masks_weighted(mask_number, mask_size, img_size, distribution):
    masks = torch.ones(mask_number, 1, img_size[0], img_size[1])
    mask_half_height, mask_half_width = mask_size[0]//2, mask_size[1]//2

    distribution = np.pad(distribution, [[mask_half_height]*2, [mask_half_width]*2], constant_values=distribution.min())
    distribution_flatten = distribution.flatten()
    distribution_normalized = distribution_flatten / distribution_flatten.sum()

    points_number = np.random.choice(
        a=np.arange(0, (img_size[0]+mask_size[0])*(img_size[1]+mask_size[1])),
        size=mask_number,
        p=distribution_normalized,
    )
    masks_center_x = points_number // (img_size[0]+mask_size[0]) - mask_half_height
    masks_center_y = points_number % (img_size[1]+mask_size[1]) - mask_half_width
    top = np.maximum(np.full_like(masks_center_x, 0), masks_center_x-mask_half_height)
    buttom = np.minimum(np.full_like(masks_center_x, img_size[0]), masks_center_x+mask_half_height)
    left = np.maximum(np.full_like(masks_center_y, 0), masks_center_y-mask_half_width)
    right = np.minimum(np.full_like(masks_center_y, img_size[1]), masks_center_y+mask_half_width)
    for i in range(mask_number):
        masks[i, :, top[i]:buttom[i], left[i]:right[i]] = 0

    return masks

def get_gradrise_map(mask_number, mask_size, input_size, distribution, layer_name, input_tensor, gradcam_sal_flatten, verbose=False):
    model = get_model()
    # Generate masks.
    masks = generate_masks_weighted(mask_number, mask_size, input_size, distribution)
    # save the sum of the masks
    mask_sum = torch.sum(masks, dim=0)[0]
    # Apply the masks on the image.
    masked_imgs = torch.mul(masks, input_tensor)
    # Put the data on the GPU.
    masked_imgs = masked_imgs.cuda()
    if verbose:
        masked_imgs = tqdm(masked_imgs)
    # Get the feature maps of the original image.
    feature = get_hidden_features(input_tensor.cuda(), layer_name, model)
    # Get features in the black box.
    masked_feature_list = []
    for masked_img in masked_imgs:
        masked_feature = get_hidden_features(masked_img[None, :, :, :], layer_name, model)
        masked_feature_list.append(masked_feature)
    masked_features = torch.cat(masked_feature_list)
    # Put the data back on the CPU.
    masked_features = masked_features.cpu()
    feature = feature.cpu()
    # Distance between the masked features and the original features.
    weights = torch.sqrt(torch.sum(torch.pow(torch.subtract(masked_features, feature), 2), dim=1))
    # Multiply the masks and the weights to get saliency maps for each feature.
    weights_reshaped = torch.flatten(weights, start_dim=1, end_dim=2).T
    masks_reshaped = 1 - torch.flatten(masks, start_dim=1, end_dim=3)
    feature_saliency_maps = torch.matmul(weights_reshaped, masks_reshaped) # (49, 50176)
    feature_saliency_maps = torch.unflatten(feature_saliency_maps, 1, (224, 224)) # (49, 224, 224)
    # Sum the saliency maps with the gradcam value as weights.
    saliency_map_weighted = torch.sum(torch.mul(feature_saliency_maps, gradcam_sal_flatten[:, None, None]), dim=0)
    # saliency_map_weighted = torch.div(saliency_map_weighted, (mask_number - mask_sum + 1))
    return saliency_map_weighted, mask_sum

def gradrise(input_tensor, input_size=(224, 224), mask_size_init=(56, 56), mask_size_final=(12, 12), N=2000, mask_decay_factor=0.9, verbose=False):
    saliency_maps = []
    mask_sums = []
    distributions = []
    mask_size = mask_size_init
    distribution = torch.ones(input_size[0], input_size[1])
    layer_name, gradcam_sal_flatten = gradcam(input_tensor, input_size=(224, 224), new_risecam=True)
    while mask_size >= mask_size_final:
        if verbose:
            print(mask_size)
        saliency_map, mask_sum = get_gradrise_map(
            N,
            mask_size,
            input_size,
            distribution,
            layer_name,
            input_tensor=input_tensor,
            gradcam_sal_flatten=gradcam_sal_flatten,
            verbose=verbose,
        )
        # weight the saliency map with the number of the pixal being selected
        saliency_map = torch.div(saliency_map, (N - mask_sum + 1))

        distribution = copy.deepcopy(saliency_map)
            
        distributions.append(distribution)
        mask_sums.append(mask_sum)

        saliency_maps.append(saliency_map)

        # sum contains Nan
        if torch.isnan(distribution).any():
            break
        # sum = 0
        if torch.sum(distribution).item() == 0:
            break
        # divide the mask size with the decay factor
        mask_size = tuple(int(size*mask_decay_factor/2) * 2 for size in mask_size)
    saliency_maps_tensor = torch.stack(saliency_maps)

    return np.array(saliency_maps_tensor.sum(dim=0))
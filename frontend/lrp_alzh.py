import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import copy
import os

class LRPVisualizer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def new_layer(self, layer, g):
        layer = copy.deepcopy(layer)
        try: layer.weight = torch.nn.Parameter(g(layer.weight))
        except AttributeError: pass
        try: layer.bias = torch.nn.Parameter(g(layer.bias))
        except AttributeError: pass
        return layer

    def dense_to_conv(self, layers):
        newlayers = []
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                newlayer = None
                if i == 0:
                    m, n = 512, layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 7)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 7, 7))
                else:
                    m, n = layer.weight.shape[1], layer.weight.shape[0]
                    newlayer = nn.Conv2d(m, n, 1)
                    newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
                newlayer.bias = nn.Parameter(layer.bias)
                newlayers += [newlayer]
            else:
                newlayers += [layer]
        return newlayers

    def get_linear_layer_indices(self):
        offset = len(self.model.vgg16._modules['features']) + 1
        indices = []
        for i, layer in enumerate(self.model.vgg16._modules['classifier']):
            if isinstance(layer, nn.Linear):
                indices.append(i)
        indices = [offset + val for val in indices]
        return indices

    def apply_lrp_on_vgg16(self, image):
        #image = torch.unsqueeze(image, 0)
        # step1 - extract the layers
        layers = list(self.model.vgg16._modules['features']) \
                    + [self.model.vgg16._modules['avgpool']] \
                    + self.dense_to_conv(list(self.model.vgg16._modules['classifier']))
        linear_layer_indices = self.get_linear_layer_indices()
        # step 2: propagate image through layers and store activations
        n_layers = len(layers)
        activations = [image] + [None] * n_layers  # list of activations

        for layer in range(n_layers):
            if layer in linear_layer_indices:
                if layer == 32:
                    activations[layer] = activations[layer].reshape((1, 512, 7, 7))
            activation = layers[layer].forward(activations[layer])
            if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
                activation = torch.flatten(activation, start_dim=1)
            activations[layer+1] = activation

        # step 3: replace last layer with one-hot encoding
        output_activation = activations[-1].detach().cpu().numpy()
        if len(output_activation.shape) > 1:
            output_activation = output_activation.flatten()  # Flatten if necessary

        max_activation = output_activation.max()
        one_hot_output = (output_activation == max_activation).astype(float)

        activations[-1] = torch.FloatTensor(one_hot_output).to(self.device)

        # step 4: backpropagate relevance scores
        relevances = [None] * n_layers + [activations[-1]]
        # Iterate over the layers in reverse order
        for layer in range(0, n_layers)[::-1]:
            current = layers[layer]
            # treat max pooling layers as avg pooling
            if isinstance(current, torch.nn.MaxPool2d):
                layers[layer] = torch.nn.AvgPool2d(2)
                current = layers[layer]
            if isinstance(current, torch.nn.Conv2d) or \
            isinstance(current, torch.nn.AvgPool2d) or\
            isinstance(current, torch.nn.Linear):
                activations[layer] = activations[layer].data.requires_grad_(True)

                # lower layers, LRP-gamma >> favor positive contributions (activations)
                if layer <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
                # middle layers, LRP-epsilon >> remove some noise / only most salient factors survive
                if 17 <= layer <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
                # upper Layers, LRP-0 >> basic rule
                if layer >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

                # transform weights of layer and execute forward pass
                z = incr(self.new_layer(layers[layer],rho).forward(activations[layer]))
                # element-wise division between relevance of the next layer and z
                s = (relevances[layer+1]/z).data
                # calculate the gradient and multiply it by the activation
                (z * s).sum().backward()
                c = activations[layer].grad
                # assign new relevance values
                relevances[layer] = (activations[layer]*c).data
            else:
                relevances[layer] = relevances[layer+1]

        # >>> potential Step 5: apply different propagation rule for pixels
        return relevances[0]

    
    def save_and_display_image(self, image):
       OUTPUT_DIR='/home/hackathon/frontend/output_LRP_ALZH'
       image_relevances = self.apply_lrp_on_vgg16(image)
       image_relevances = image_relevances.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
       image_relevances = np.interp(image_relevances, (image_relevances.min(), image_relevances.max()), (0, 1))
       plt.figure(figsize=(10, 5))
       plt.subplot(1, 2, 1)
       plt.axis('off')
       plt.imshow(image_relevances[:, :, 0], cmap="seismic")

       plt.subplot(1, 2, 2)
       plt.axis('off')
       plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    
       output_path = os.path.join(OUTPUT_DIR, "output_image.png")
       plt.savefig(output_path)
       plt.close()
    
       return output_path
    
    

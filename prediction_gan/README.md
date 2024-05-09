# Prediction Optimizer (to stabilize GAN training)

### Introduction
This is a PyTorch implementation of 'prediction method' introduced in the following paper ...

- Abhay Yadav et al., Stabilizing Adversarial Nets with Prediction Methods, ICLR 2018, [Link](https://openreview.net/forum?id=Skj8Kag0Z&noteId=rkLymJTSf)
- (*Just for clarification, I'm not an author of the paper.*)

The authors proposed a simple (but effective) method to stabilize GAN trainings. With this Prediction Optimizer, you can easily apply the method to your existing GAN codes. This impl. is compatible with **most of PyTorch optimizers and network structures**. (Please let me know if you have any issues using this)


### How-to-use

#### Instructions
  - Import prediction.py
    - `from prediction import PredOpt`
  - Initialize just like an optimizer
    - `pred = PredOpt(net.parameters())`
  - Run the model in a 'with' block to get results from a model with predicted params.
    - With 'step' argument, you can control lookahead step size (1.0 by default)
    - ```python
      with pred.lookahead(step=1.0):
          output = net(input)
      ``` 
  - Call step() after an update of the network parameters
    - ```python
      optim_net.step()
      pred.step()
      ```

#### Examples
  - A example snippet
  - ```python
    import torch.optim as optim
    from prediction import PredOpt
    
    
    # ...
    
    optim_G = optim.Adam(netG.parameters(), lr=0.01)
    optim_D = optim.Adam(netD.parameters(), lr=0.01)
    
    pred_G = PredOpt(netG.parameters())             # Create an prediction optimizer with target parameters
    pred_D = PredOpt(netD.parameters())
    
    
    for i, data in enumerate(dataloader, 0):
        # (1) Training D with samples from predicted generator
        with pred_G.lookahead(step=1.0):            # in the 'with' block, the model works as a 'predicted' model
            fake_predicted = netG(Z)
            
            # Compute gradients and loss 
            
            optim_D.step()
            pred_D.step()
        
        # (2) Training G
        with pred_D.lookahead(step=1.0):            # 'Predicted D'
            fake = netG(Z)                          # Draw samples from the real model. (not predicted one)
            D_outs = netD(fake)
            
            # Compute gradients and loss
            
            optim_G.step()
            pred_G.step()                           # You should call PredOpt.step() after each update
    ``` 

### External links

- GitHub repo. mentioned in the paper (https://github.com/jaiabhayk/stableGAN)
  - Empty by the date of this README.md update.
- Another impl. for PyTorch (https://github.com/shahsohil/stableGAN)
  - From the name of the repository owner, I guess it's written by one of the paper authors. (not 100% sure)
  - Currently supports Adam only.

